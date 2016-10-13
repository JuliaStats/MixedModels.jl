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
    ranef!{T}(v::Vector{Matrix{T}}, m::MixedModel{T}, uscale::Bool = false)

Overwrites `v` with the conditional modes of the random effects for `m`.

If `uscale` is `true` the random effects are on the spherical (i.e. `u`) scale, otherwise on the
original scale
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
        LinAlg.A_ldiv_B!(isa(Rjj, Diagonal) ? Rjj : UpperTriangular(Rjj), uj)
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
        setnames!(vnm, levs(trm.f), 2)
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
