
"""
cond(m::MixedModel)

Return a vector of condition numbers of the λ matrices for the random-effects terms
"""
LinearAlgebra.cond(m::MixedModel) = cond.(m.λ)

"""
    issingular(m::MixedModel, θ=m.θ)

Test whether the model `m` is singular if the parameter vector is `θ`.

Equality comparisons are used b/c small non-negative θ values are replaced by 0 in `fit!`.
"""
issingular(m::MixedModel, θ=m.θ) = any(lowerbd(m) .== θ)

# FIXME: better to base this on m.optsum.returnvalue
StatsBase.isfitted(m::MixedModel) = m.optsum.feval > 0

StatsBase.model_response(m::MixedModel) = m.y

StatsBase.nobs(m::MixedModel) = length(m.y)

function retbl(mat, trm)
    merge(
        NamedTuple{(fname(trm),)}((trm.levels,)),
        columntable(Tables.table(transpose(mat), header=Symbol.(trm.cnames))),
        )
end

"""
    raneftables(m::LinearMixedModel; uscale = false)

Return the conditional means of the random effects as a NamedTuple of columntables
"""
function raneftables(m::MixedModel{T}; uscale = false) where {T}
    NamedTuple{fnames(m)}((map(retbl, ranef(m, uscale=uscale), m.reterms)...,))
end

function σs(m::MixedModel)
    σ = dispersion(m)
    NamedTuple{fnames(m)}(((σs(t, σ) for t in m.reterms)...,))
end

function σρs(m::MixedModel)
    σ = dispersion(m)
    NamedTuple{fnames(m)}(((σρs(t, σ) for t in m.reterms)...,))
end

"""
    size(m::MixedModel)

Returns the size of a mixed model as a tuple of length four:
the number of observations, the number of (non-singular) fixed-effects parameters,
the number of conditional modes (random effects), the number of grouping variables
"""
function Base.size(m::MixedModel)
    dd = m.dims
    dd.n, dd.p, sum(size.(m.reterms, 2)), dd.nretrms
end

"""
    vcov(m::MixedModel; corr=false)

Returns the variance-covariance matrix of the fixed effects.
If `corr` is `true`, the correlation of the fixed effects is returned instead.
"""
function StatsBase.vcov(m::MixedModel; corr=false)
    Xtrm = m isa GeneralizedLinearMixedModel ? m.LMM.feterm : m.feterm
    iperm = invperm(Xtrm.piv)
    p = length(iperm)
    r = Xtrm.rank
    Linv = inv(feL(m))
    T = eltype(Linv)
    permvcov = dispersion(m, true) * (Linv'Linv)
    if p == Xtrm.rank
        vv = permvcov[iperm, iperm]
    else
        covmat = fill(zero(T) / zero(T), (p, p))
        for j = 1:r, i = 1:r
            covmat[i, j] = permvcov[i, j]
        end
        vv = covmat[iperm, iperm]
    end

    corr ?  StatsBase.cov2cor!(vv, stderror(m)) : vv
end
