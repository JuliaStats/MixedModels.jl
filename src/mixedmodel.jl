#  Functions and methods common to all MixedModel types

"""
    feL(m::MixedModel)

Return the lower Cholesky factor for the fixed-effects parameters, as an `LowerTriangular`
`p × p` matrix.
"""
feL(m::MixedModel) = LowerTriangular(lmm(m).L.data.blocks[end - 1, end - 1])

lmm(m::LinearMixedModel) = m
lmm(m::GeneralizedLinearMixedModel) = m.LMM

"""
    cond(m::MixedModel)

Return the vector of the condition numbers of the blocks of `m.trms`
"""
LinearAlgebra.cond(m::MixedModel) = cond.(reterms(m))

"""
    std{T}(m::MixedModel{T})

Return the estimated standard deviations of the variance components as a `Vector{Vector{T}}`.
"""
function Statistics.std(m::MixedModel{T}) where {T}
    rl =  filter(!isempty, rowlengths.(lmm(m).trms))
    s = sdest(m)
    isfinite(s) ? rmul!(push!(rl, [1.]), s) : rl
end

fixefnames(m::MixedModel) = lmm(m).trms[end - 1].cnames
## methods for generics defined in StatsBase

function StatsBase.coeftable(m::MixedModel)
    co = coef(m)
    se = stderror(m)
    z = co ./ se
    pvalue = ccdf.(Chisq(1), abs2.(z))
    CoefTable(hcat(co, se, z, pvalue), ["Estimate", "Std.Error", "z value", "P(>|z|)"],
        fixefnames(m), 4)
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
    for i in 1 : nblocks(A, 2), j in 1 : i
        println(io, i, ",", j, ": ", typeof(A[Block(i, j)]), " ",
                blocksize(A, (i, j)), " ", typeof(L.data[Block(i, j)]))
    end
end
describeblocks(m::MixedModel) = describeblocks(stdout, m)

"""
    fnames(m::MixedModel)

Return the names of the grouping factors for the random-effects terms.
"""
fnames(m::MixedModel) = map(x -> x.fnm, reterms(m))

function getθ!(v::AbstractVector{T}, m::LinearMixedModel{T}) where {T}
    trms = m.trms
    @argcheck(length(v) == sum(nθ, trms), DimensionMismatch)
    offset = 0
    for λ in trms
        nli = nθ(λ)
        getθ!(view(v, offset .+ (1 : nli)), λ)
        offset += nli
    end
    v
end

"""
    getΛ(m::MixedModel)

Return a vector of covariance template matrices for the random effects of `m`
"""
getΛ(m::MixedModel) = getΛ.(reterms(m))

"""
    getθ(m::MixedModel)

Return the current covariance parameter vector.
"""
getθ(m::MixedModel) = getθ(lmm(m).trms)

"""
    grplevels(m::MixedModel)

Return the number of levels in the random-effects terms' grouping factors.
"""
grplevels(m::MixedModel) = nlevs.(reterms(m))

nreterms(m::MixedModel) = sum(t -> isa(t, AbstractFactorReTerm), lmm(m).trms)

reterms(m::MixedModel) = convert(Vector{AbstractFactorReTerm},
                                 filter(t -> isa(t, AbstractFactorReTerm), lmm(m).trms))

"""
    ranef!{T}(v::Vector{Matrix{T}}, m::MixedModel{T}, β, uscale::Bool)

Overwrite `v` with the conditional modes of the random effects for `m`.

If `uscale` is `true` the random effects are on the spherical (i.e. `u`) scale, otherwise
on the original scale
"""
function ranef!(v::Vector, m::LinearMixedModel{T}, β::AbstractArray{T}, uscale::Bool) where {T}
    Ldat = m.L.data
    @argcheck((k = length(v)) == nreterms(m), DimensionMismatch)
    for j in 1:k
        mulαβ!(vec(copyto!(v[j], Ldat[Block(nblocks(Ldat, 2), j)])), Ldat[Block(k + 1, j)]', β,
            -one(T), one(T))
    end
    for i in k: -1 :1
        Lii = Ldat[Block(i, i)]
        vi = vec(v[i])
        ldiv!(adjoint(isa(Lii, Diagonal) ? Lii : LowerTriangular(Lii)), vi)
        for j in 1:(i - 1)
            mulαβ!(vec(v[j]), Ldat[Block(i, j)]', vi, -one(T), one(T))
        end
    end
    if !uscale
        trms = m.trms
        for (j, vv) in enumerate(v)
            lmul!(trms[j].Λ, vv)
        end
    end
    v
end

ranef!(v::Vector, m::LinearMixedModel, uscale::Bool) = ranef!(v, m, fixef(m), uscale)

"""
    ranef(m::MixedModel; uscale=false, named=true)

Return, as a `Vector{Vector{T}}` (`Vector{NamedVector{T}}` if `named=true`),
the conditional modes of the random effects in model `m`.

If `uscale` is `true` the random effects are on the spherical (i.e. `u`) scale, otherwise on
the original scale.
"""
function ranef(m::MixedModel; uscale=false, named=false)
    LMM = lmm(m)
    T = eltype(LMM.sqrtwts)
    retrms = reterms(LMM)
    v = Matrix{T}[Matrix{T}(undef, vsize(t), nlevs(t)) for t in retrms]
    ranef!(v, LMM, uscale)
    named || return v
    vnmd = map(NamedArray, v)
    for (trm, vnm) in zip(retrms, vnmd)
        setnames!(vnm, trm.cnms, 1)
        setnames!(vnm, string.(levs(trm)), 2)
    end
    vnmd
end

function StatsBase.vcov(m::MixedModel)
    Xtrm = lmm(m).trms[end - 1]
    iperm = invperm(Xtrm.piv)
    p = length(iperm)
    r = Xtrm.rank
    Linv = inv(feL(m))
    permvcov = varest(m) * (Linv'Linv)
    if p == Xtrm.rank
        permvcov[iperm, iperm]
    else
        T = eltype(permvcov)
        covmat = fill(zero(T)/zero(T), (p, p))
        for j in 1:r, i in 1:r
            covmat[i,j] = permvcov[i, j]
        end
        covmat[iperm, iperm]
    end
end

Statistics.cor(m::MixedModel{T}) where {T} = Matrix{T}[stddevcor(t)[2] for t in reterms(m)]

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

FIXME: Change the output type to that of the (1,1) block of lmm(m).L.data
"""
function condVar(m::MixedModel)
    nreterms(m) == 1 || throw(ArgumentError("condVar requires a single r.e. term"))
    condVar(lmm(m).L.data[Block(1,1)], first(lmm(m).trms).Λ, varest(m))
end

function condVar(L11::Diagonal{T}, Λ::T, ssqr::T) where {T}
    Ld = L11.diag
    Array{T, 3}[reshape(abs2.(Λ ./ Ld) .* ssqr, (1, 1, :))]
end

function condVar(L11::UniformBlockDiagonal{T}, Λ::LowerTriangular{T}, ssqr::T) where {T}
    value = copy(L11)
    scratch = similar(first(L11.facevec))
    for L in value.facevec
        LinearAlgebra.inv!(LowerTriangular(copyto!(scratch, LowerTriangular(L))))
        rmul!(scratch, Λ)
        rmul!(mul!(L, scratch', scratch), ssqr)
    end
    value
end