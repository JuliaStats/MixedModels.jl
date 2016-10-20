#  Functions and methods common to all MixedModel types

"""
    feR(m::MixedModel)

Return th upper Cholesky factor for the fixed-effects parameters, as an `UpperTriangular`
`p × p` matrix.
"""
function feR(m::MixedModel)
    R = lmm(m).R
    kp1 = size(R, 1) - 1
    UpperTriangular(R[kp1, kp1])
end

"""
    lmm(m::MixedModel)

Extract the `LinearMixedModel` from a `MixedModel`.

If `m` is a `LinearMixedModel` return `m`. If `m` is a
`GeneralizedLinearMixedModel` return `m.LMM`.
"""
lmm(m::LinearMixedModel) = m
lmm(m::GeneralizedLinearMixedModel) = m.LMM

"""
    cond(m::MixedModel)

Returns the vector of the condition numbers of the blocks of `m.Λ`
"""
Base.cond(m::MixedModel) = [cond(λ)::Float64 for λ in lmm(m).Λ]

"""
    std{T}(m::MixedModel{T})

The estimated standard deviations of the variance components as a `Vector{Vector{T}}`.
"""
Base.std(m::MixedModel) = sdest(m) * push!([rowlengths(λ) for λ in lmm(m).Λ], [1.])

## methods for generics defined in StatsBase

function StatsBase.coeftable(m::MixedModel)
    fe = fixef(m)
    se = stderr(m)
    z = fe ./ se
    pvalue = ccdf(Chisq(1), abs2(z))
    CoefTable(hcat(fe, se, z, pvalue), ["Estimate", "Std.Error", "z value", "P(>|z|)"],
        coefnames(lmm(m).mf), 4)
end

"""
    describeblocks(io::IO, m::MixedModel)
    describeblocks(m::MixedModel)

Describe the types and sizes of the blocks in the upper triangle of `m.A` and `m.R`.
"""
function describeblocks(io::IO, m::MixedModel)
    lm = lmm(m)
    A, R = lm.A, lm.R
    for j in 1:size(A,2), i in 1:j
        println(io, i, ",", j, ": ", typeof(A[i,j]), " ", size(A[i,j]), " ", typeof(R[i,j]))
    end
    nothing
end
describeblocks(m::MixedModel) = describeblocks(Base.STDOUT, m)

"""
    fnames(m::MixedModel)

Returns the names of the grouping factors for the random-effects terms.
"""
function fnames(m::MixedModel)
    lm = lmm(m)
    [t.fnm for t in lm.wttrms[1:length(lm.Λ)]]
end

function getθ!(v::AbstractVector, m::MixedModel)
    Λ = lmm(m).Λ
    nl = map(nlower, Λ)
    if length(v) ≠ sum(nl)
        throw(DimensionMismatch("length(v) = $(length(v)) should be $(sum(nl))"))
    end
    offset = 0
    for i in eachindex(nl)
        nli = nl[i]
        getθ!(view(v, offset + (1 : nli)), Λ[i])
        offset += nli
    end
    v
end

function getθ(m::MixedModel)
    Λ = lmm(m).Λ
    getθ!(Array(eltype(Λ[1]), sum(A -> nlower(A), Λ)), m)
end

"""
    grplevels(m::MixedModel)

Returns the number of levels in the random-effects terms' grouping factors.
"""
function grplevels(m::MixedModel)
    lm = lmm(m)
    [length(lm.trms[i].f.pool) for i in eachindex(lm.Λ)]
end

"""
    ranef!{T}(v::Vector{Matrix{T}}, m::MixedModel{T}, β, uscale::Bool)

Overwrites `v` with the conditional modes of the random effects for `m`.

If `uscale` is `true` the random effects are on the spherical (i.e. `u`) scale, otherwise on the
original scale
"""
function ranef!{T}(v::Vector, m::LinearMixedModel{T}, β::AbstractArray{T}, uscale::Bool)
    R, Λ = m.R, m.Λ
    k = length(Λ)        # number of random-effects terms
    for j in 1:k
        copy!(v[j], R[j, end])
    end
    kp1 = k + 1
    if !isempty(β)       #  in the pirls! function for GLMMs want to skip this
        for j in 1:k     # subtract the fixed-effects contribution
            BLAS.gemv!('N', -1.0, R[j, kp1], β, 1.0, vec(v[j]))
        end
    end
    for j in k:-1:1
        Rjj = R[j, j]
        uj = vec(v[j])
        LinAlg.A_ldiv_B!(isa(Rjj, Diagonal) ? Rjj : UpperTriangular(Rjj), uj)
        for i in 1:(j - 1)
            Rij = R[i, j]
            if isa(Rij, StridedMatrix{T})
                BLAS.gemv!('N', -one(T), R[i, j], uj, one(T), vec(v[i]))
            else
                A_mul_B!(-one(T), R[i, j], uj, one(T), vec(v[i]))
            end
        end
    end
    if !uscale
        for j in 1:k
            A_mul_B!(Λ[j], v[j])
        end
    end
    v
end

function ranef!(v::Vector, m::LinearMixedModel, uscale::Bool)
    ranef!(v, m, feR(m) \ vec(m.R[end-1, end]), uscale)
end

"""
    ranef{T}(m::MixedModel{T}, uscale=false)

Returns, as a `Vector{Matrix{T}}`, the conditional modes of the random effects in model `m`.

If `uscale` is `true` the random effects are on the spherical (i.e. `u`) scale, otherwise on the
original scale.
"""
function ranef(m::MixedModel; uscale=false, named=false)
    lm = lmm(m)
    Λ, trms = lm.Λ, lm.trms
    T = eltype(trms[end])
    v = Matrix{T}[]
    for i in eachindex(Λ)
        l = size(Λ[i], 1)
        k = size(trms[i], 2)
        push!(v, Array(T, (l, div(k, l))))
    end
    ranef!(v, lm, uscale)
    named || return v
    vnmd = NamedArray.(v)
    for (trm, vnm) in zip(trms, vnmd)
        setnames!(vnm, trm.cnms, 1)
        setnames!(vnm, string.(levs(trm)), 2)
    end
    vnmd
end

"""
     vcov(m::MixedModel)

Returns the estimated covariance matrix of the fixed-effects estimator.
"""
function StatsBase.vcov(m::MixedModel)
    Rinv = inv(feR(m))
    varest(m) * (Rinv * Rinv')
end

"""
    cor(m::MixedModel)

Returns the estimated correlation matrix of the fixed-effects estimator.
"""
function StatsBase.cor(m::MixedModel)
    vc = vcov(m)
    scl = [√(inv(vc[i])) for i in diagind(vc)]
    scale!(scl, scale!(vc, scl))
end

function convert(::Type{LinAlg.Cholesky}, m::MixedModel)
    R = lmm(m).R
    nblk = size(R, 2) - 1
    sizes = [size(R[1, j], 2) for j in 1 : nblk]
    offsets = unshift!(cumsum(sizes), 0)
    res = zeros(eltype(R[1, end]), (offsets[end], offsets[end]))
    for j in 1 : nblk
        jinds = (1 : sizes[j]) + offsets[j]
        for i in 1 : j
            copy!(view(res, (1 : sizes[i]) + offsets[i], jinds), R[i, j])
        end
    end
    LinAlg.Cholesky(res, 'U')
end

"""
    condVar(m::MixedModel)

Return the conditional variances matrices of the random effects.

The random effects are returned by `ranef` as a vector of length `k`,
where `k` is the number of random effects terms.  The `i`th element
is a matrix of size `vᵢ × ℓᵢ`  where `vᵢ` is the size of the
vector-valued random effects for each of the `ℓᵢ` levels of the grouping
factor.  Technically those values are the modes of the conditional
distribution of the random effects given the observed data.

This function returns an array of `k` three dimensional arrays,
where the `i`th array is of size `vᵢ × vᵢ × ℓᵢ`.  These are the
diagonal blocks from the conditional variance-covariance matrix,

    s² Λ(Λ'Z'ZΛ + I)Λ'
"""
function condVar(m::MixedModel)
    lm = lmm(m)
    Λ = lm.Λ
    if length(Λ) > 1
        throw(ArgumentError(
            "code for more than one term not yet written"))
    end
    A = lm.A[1,1]
    res = Array{eltype(A),3}[]
    if isa(A, Diagonal)
        push!(res, reshape(inv.(A.diag) .* abs2(Λ[1][1]), (1,1,size(A,1))))
    else
        throw(ArgumentError(
            "code for vector-value random-effects not yet written"))
    end
    res
end
