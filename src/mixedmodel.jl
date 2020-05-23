
"""
cond(m::MixedModel)

Return a vector of condition numbers of the λ matrices for the random-effects terms
"""
LinearAlgebra.cond(m::MixedModel) = cond.(m.λ)

function σs(m::MixedModel)
    σ = dispersion(m)
    NamedTuple{fnames(m)}(((σs(t, σ) for t in m.reterms)...,))
end

function σρs(m::MixedModel)
    σ = dispersion(m)
    NamedTuple{fnames(m)}(((σρs(t, σ) for t in m.reterms)...,))
end

"""
    vcov(m::LinearMixedModel)

Returns the variance-covariance matrix of the fixed effects.
If `corr=true`, then correlation of fixed effects is returned instead.
"""
function StatsBase.vcov(m::MixedModel; corr=false)
    Xtrm = fetrm(m)
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
