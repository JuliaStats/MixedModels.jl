module MixedModelsVcovExt

using MixedModels
using MixedModels: StatsAPI
using Vcov: Vcov, CovarianceEstimator, VcovData


# in 0.8, Vcov uses invcrossmodelmatrix as an accessor function
# for a precomputed inv(crossmodelmatrix(m))
if pkgversion(Vcov) >= v"0.8"
    Vcov.invcrossmodelmatrix(m::MixedModel) = inv(crossmodelmatrix(m))

    function Vcov.VcovData(m::MixedModel)
        cm = crossmodelmatrix(m)
        ddf = nobs(m) - round(Int, sum(leverage(m)))
        return VcovData(modelmatrix(m), cm, inv(cm), residuals(m), ddf)
    end

end

# to avoid method ambiguities, we define this separately for each
# relevant CovarianceEstimator
StatsAPI.vcov(m::MixedModel, c::Vcov.SimpleCovariance) = vcov(VcovData(m), c)
StatsAPI.vcov(m::MixedModel, c::Vcov.RobustCovariance) = vcov(VcovData(m), c)

# Vcov.vcov(m::MixedModel, ::Vcov.SimpleCovariance) = vcov(m)

end # module
