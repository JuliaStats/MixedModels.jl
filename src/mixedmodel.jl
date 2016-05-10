#  Functions and methods common to all MixedModel types

"""
    lmm(m::MixedModel)

Extract the `LinearMixedModel` from a `MixedModel`.  If `m` is itself a `LinearMixedModel` this simply returns `m`.
If `m` is a `GeneralizedLinearMixedModel` this returns its `LMM` member.

Args:

- `m`: a `MixedModel`

Returns:
  A `LinearMixedModel`, either `m` itself or the `LMM` member of `m`
"""
lmm(m::LinearMixedModel) = m

## Methods for generics defined in Base.

"""
    cond(m::MixedModel)

Args:

- `m`: a `MixedModel`

Returns:
  A `Vector` of the condition numbers of the blocks of `m.Λ`
"""
Base.cond(m::MixedModel) = [cond(λ)::Float64 for λ in lmm(m).Λ]

Base.getindex(m::MixedModel, s::Symbol) = mapreduce(x->x[s], vcat, lmm(m).Λ)

function Base.setindex!(m::MixedModel, v::Vector, s::Symbol)
    if s ≠ :θ
        throw(ArgumentError("only ':θ' is meaningful for assignment"))
    end
    lm = lmm(m)
    lam = lm.Λ
    if length(v) != sum(nlower, lam)
        throw(DimensionMismatch("length(v) = $(length(v)), should be $(sum(nlower, lam))"))
    end
    A, R = lm.A, lm.R
    n = size(A, 1)                      # inject upper triangle of A into L
    for j in 1:n, i in 1:j
        inject!(R[i,j],A[i,j])
    end
    offset = 0
    for i in eachindex(lam)
        li = lam[i]
        nti = nlower(li)
        li[:θ] = sub(v, offset + (1:nti))
        offset += nti
        for j in i:size(R,2)
            tscale!(li, R[i,j])
        end
        for ii in 1:i
            tscale!(R[ii,i], li)
        end
        inflate!(R[i,i])
    end
    cfactor!(R)
end

"""
    std(m)

Estimated standard deviations of the variance components

Args:

- `m`: a `MixedModel`

Returns:
  `Vector{Vector{Float64}}`
"""
Base.std(m::MixedModel) = sdest(m) * push!([rowlengths(λ) for λ in lmm(m).Λ], [1.])

## methods for generics defined in StatsBase

StatsBase.coef(m::LinearMixedModel) = fixef(m)

StatsBase.loglikelihood(m::MixedModel) = -deviance(m)/2

StatsBase.nobs(m::LinearMixedModel) = length(model_response(m))

## methods for functions exported from this module

function StatsBase.coeftable(m::MixedModel)
    fe = fixef(m)
    se = stderr(m)
    CoefTable(hcat(fe, se, fe ./ se), ["Estimate", "Std.Error", "z value"], coefnames(lmm(m).mf))
end

"""
    describeblocks(io, m)
    describeblocks(m)

Describe the blocks of the A and R matrices

Args:

- `m`: a `MixedModel`

Prints a description of the types and sizes of the blocks in `A`, the matrix of
products of terms, and in `R`, the upper Cholesky factor.
"""
function describeblocks(io::IO, m::MixedModel)
    lm = lmm(m)
    A, R = lm.A, lm.R
    for j in 1:size(A,2), i in 1:j
        println(io, i,",",j,": ",typeof(A[i,j])," ",size(A[i,j])," ",typeof(R[i,j]))
    end
    nothing
end
describeblocks(m::MixedModel) = describeblocks(Base.STDOUT, m)

"""
    feR(m)

Upper Cholesky factor for the fixed-effects parameters

Args:

- `m`: a `MixedModel`

Returns:
  an `UpperTriangular` p × p matrix which is the upper Cholesky factor
"""
function feR(m::MixedModel)
    lm = lmm(m)
    kp1 = length(lm.Λ) + 1
    UpperTriangular(lm.R[kp1, kp1])
end

"""
    fnames(m)

Args:

- `m`: a `MixedModel`

Returns:
  A `Vector{AbstractString}` of names of the grouping factors for the random-effects terms.
"""
function fnames(m::MixedModel)
    lm = lmm(m)
    [t.fnm for t in lm.wttrms[1:length(lm.Λ)]]
end

"""
    grplevels(m)

Args:

- `m`: a `MixedModel`

Returns:
 A `Vector{Int}` containing the number of levels in each term's grouping factor
"""
grplevels(m::MixedModel) = [length(t.f.pool) for t in reterms(m)]

"""
    lowerbd(m)

Args:

- `m`: a `MixedModel`

Returns:
  A `Vector` of lower bounds on the covariance parameter vector `m[:θ]`
"""
lowerbd(m::MixedModel) = mapreduce(lowerbd,vcat,lmm(m).Λ)

"""
    ranef!(v, m, uscale)

Overwrite v with the conditional modes of the random effects for `m`

Args:

- `v`: a `Vector` of matrices
- `m`: a `MixedModel`
- `uscale`: `Bool`, return the random-effects on the spherical (i.e. `u`) scale?

Returns:
  `v`, overwritten with the conditional modes
"""
function ranef!(v::Vector, m::MixedModel, uscale)
    R, Λ = m.R, m.Λ
    k = length(Λ)        # number of random-effects terms
    for j in 1:k
        copy!(v[j], R[j, end])
    end
    rβ = R[k + 1, end]
    if !isempty(rβ)      #  in the pirls! function for GLMMs want to skip this
        β = vec(feR(m) \ rβ)
        kp1 = k + 1
        for j in 1:k     # subtract the fixed-effects contribution
            BLAS.gemv!('N', -1.0, R[j, kp1], β, 1.0, vec(v[j]))
        end
    end
    for j in k:-1:1
        Rjj = R[j, j]
        uj = vec(v[j])
        Base.LinAlg.A_ldiv_B!(isa(Rjj, Diagonal) ? Rjj : UpperTriangular(Rjj), uj)
        for i in 1:j - 1
            ui = vec(v[i])
            ui -= R[i, j] * uj
        end
    end
    if !uscale
        for j in 1:k
            A_mul_B!(Λ[j], v[j])
        end
    end
    v
end

"""
    ranef(m)
    ranef(m, uscale)

Conditional modes of the random effects in model `m`

Args:

- `m`: a fitted `MixedModel` object
- `uscale`: a `Bool` indicating conditional modes are on the `u` scale or the `b` scale.  Defaults to `false`

Returns:
  A `Vector` of matrices of the conditional modes of the random effects on the indicated scale.
  For a scalar random-effects term the matrix is `1 × k` where `k` is the number of levels of the grouping factor.
  For a vector-valued random-effects term the matrix is `l × k` where `l` is the dimension of each random effect.
"""
function ranef(m::MixedModel, uscale=false)
    lm = lmm(m)
    Λ, trms = lm.Λ, unwttrms(lm)
    T = eltype(trms[end])
    v = Matrix{T}[]
    for i in eachindex(Λ)
        l = size(Λ[i], 1)
        k = size(trms[i], 2)
        push!(v, Array(T, (l, div(k, l))))
    end
    ranef!(v, lm, uscale)
end

"""
    reterms(m)

Args:

- `m`: a `MixedModel`

Returns:
   A `Vector` of random-effects terms.
"""
function reterms(m::MixedModel)
    lm = lmm(m)
    unwttrms(m)[1:length(lm.Λ)]
end

"""
     vcov(m)

Estimated covariance matrix of the fixed-effects estimator

Args:

- `m`: a `LinearMixedModel`

Returns
  a `p × p` `Matrix`
"""
function StatsBase.vcov(m::MixedModel)
    Rinv = Base.LinAlg.inv(feR(m))
    varest(m) * Rinv * Rinv'
end
