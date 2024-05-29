module MixedModelsVcovExt

using MixedModels
using MixedModels: StatsAPI
using MixedModels.StatsModels: hasintercept

using Vcov: Vcov, CovarianceEstimator, VcovData

function Vcov.VcovData(m::MixedModel)
    cm = crossmodelmatrix(m)
    mm = modelmatrix(m)
    # from just the fixed effects
    ddf = nobs(m) - rank(m) - hasintercept(formula(m)) # dof_residual(m) # nobs(m) - round(Int, sum(leverage(m)))
    resid = response(m) - mm * coef(m) # residuals(m)
    @static if pkgversion(Vcov) >= v"0.8"
        # Vcov 0.8 stores the inverse of crossmodelmatrix
        return VcovData(mm, cm, inv(cm), resid, ddf)
    else
        return VcovData(mm, cm, resid, ddf)
    end

end

# to avoid method ambiguities, we define this separately for each
# relevant CovarianceEstimator
StatsAPI.vcov(m::MixedModel, c::Vcov.SimpleCovariance) = vcov(VcovData(m), c)
StatsAPI.vcov(m::MixedModel, c::Vcov.RobustCovariance) = vcov(VcovData(m), c)
# StatsAPI.vcov(m::MixedModel, c::Vcov.ClusterCovariance) = vcov(VcovData(m), c)

end # module
