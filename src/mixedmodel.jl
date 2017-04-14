#  Functions and methods common to all MixedModel types

"""
    feL(m::MixedModel)

Return the lower Cholesky factor for the fixed-effects parameters, as an `LowerTriangular`
`p × p` matrix.
"""
function feL(m::MixedModel)
    L = lmm(m).L
    kp1 = size(L, 1) - 1
    LowerTriangular(L[kp1, kp1])
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
    pvalue = ccdf(Chisq(1), abs2.(z))
    CoefTable(hcat(fe, se, z, pvalue), ["Estimate", "Std.Error", "z value", "P(>|z|)"],
        lmm(m).fixefnames, 4)
end

"""
    describeblocks(io::IO, m::MixedModel)
    describeblocks(m::MixedModel)

Describe the types and sizes of the blocks in the lower triangle of `m.A` and `m.L`.
"""
function describeblocks(io::IO, m::MixedModel)
    lm = lmm(m)
    A = lm.A
    L = lm.L
    for i in 1 : size(A, 2), j in 1 : i
        println(io, i, ",", j, ": ", typeof(A[i,j]), " ", size(A[i,j]), " ", typeof(L[i,j]))
    end
end
describeblocks(m::MixedModel) = describeblocks(Base.STDOUT, m)

"""
    fnames(m::MixedModel)

Returns the names of the grouping factors for the random-effects terms.
"""
fnames(m::MixedModel) = map(x -> x.fnm, reterms(m))

function getθ!{T}(v::AbstractVector{T}, m::LinearMixedModel{T})
    Λ = lmm(m).Λ
    nl = map(nθ, Λ)
    @argcheck length(v) == sum(nl) DimensionMismatch
    offset = 0
    for i in eachindex(nl)
        nli = nl[i]
        getθ!(view(v, offset + (1 : nli)), Λ[i])
        offset += nli
    end
    v
end

getΘ{T}(m::GeneralizedLinearMixedModel{T}) = getΘ(m.LMM)
getθ{T}(m::LinearMixedModel{T}) = getθ!(Array{T}(sum(A -> nθ(A), m.Λ)), m)

"""
    grplevels(m::MixedModel)

Returns the number of levels in the random-effects terms' grouping factors.
"""
grplevels(m::MixedModel) = nlevs.(reterms(m))

nreterms(m::MixedModel) = sum(t -> isa(t, ReMat), lmm(m).trms)

reterms(m::MixedModel) = convert(Vector{ReMat}, filter(t -> isa(t, ReMat), lmm(m).trms))

"""
    ranef!{T}(v::Vector{Matrix{T}}, m::MixedModel{T}, β, uscale::Bool)

Overwrites `v` with the conditional modes of the random effects for `m`.

If `uscale` is `true` the random effects are on the spherical (i.e. `u`) scale, otherwise on the
original scale
"""
function ranef!{T}(v::Vector, m::LinearMixedModel{T}, β::AbstractArray{T}, uscale::Bool)
    L = m.L
    Λ = m.Λ
    if (k = length(v)) ≠ length(Λ)
        throw(DimensionMismatch("length(v) = $(length(v)), should be $(length(Λ))"))
    end
    for j in 1:k
        Ac_mul_B!(-one(T), L[k + 1, j], β, one(T), vec(copy!(v[j], L[end, j])))
    end
    for i in k: -1 :1
        Lii = L[i, i]
        vi = vec(v[i])
        Ac_ldiv_B!(isa(Lii, Diagonal) ? Lii : LowerTriangular(Lii), vi)
        for j in 1:(i - 1)
            Ac_mul_B!(-one(T), L[i, j], vi, one(T), vec(v[j]))
        end
    end
    if !uscale
        for j in 1:k
            A_mul_B!(Λ[j], vec(v[j]))
        end
    end
    v
end

function ranef!(v::Vector, m::LinearMixedModel, uscale::Bool)
    ranef!(v, m, Ac_ldiv_B(feL(m), vec(copy(m.L[end, end - 1]))), uscale)
end

"""
    ranef{T}(m::MixedModel{T}, uscale=false)

Returns, as a `OrderedDict{}`, the conditional modes of the random effects in model `m`.

If `uscale` is `true` the random effects are on the spherical (i.e. `u`) scale, otherwise on the
original scale.
"""
function ranef(m::MixedModel; uscale=false, named=false)
    LMM = lmm(m)
    trms = LMM.trms
    T = eltype(trms[end])
    v = Matrix{T}[]
    for trm in filter(t -> isa(t, ReMat), trms)
        push!(v, Array{T}((vsize(trm), nlevs(trm))))
    end
    ranef!(v, LMM, uscale)
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
    Linv = inv(feL(m))
    varest(m) * (Linv'Linv)
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

    s² Λ(Λ'Z'ZΛ + I)⁻¹Λ'
"""
function condVar(m::MixedModel)
    lm = lmm(m)
    Λ = lm.Λ
    if length(Λ) > 1
        throw(ArgumentError("code for more than one term not yet written"))
    end
    λ = Λ[1].m.data
    L = lm.L[1,1]
    res = Array{eltype(λ),3}[]
    if isa(L, Diagonal) && all(d -> size(d) == (1,1), L.diag) && length(λ) == 1
        l1 = λ[1]
        Ld = L.diag
        push!(res, reshape([abs2(l1 / d[1]) for d in Ld], (1, 1, length(Ld))))
    else
        throw(ArgumentError("code for vector-value random-effects not yet written"))
    end
    res *= varest(m)
end
