module MixedModelsVcovExt

using LinearAlgebra: Symmetric, inv, rank
using MixedModels: MixedModels, GeneralizedLinearMixedModel, LinearMixedModel, MixedModel
using MixedModels: feL
using StatsAPI: StatsAPI, nobs, residuals, vcov
using Vcov: Vcov, VcovData

#####
##### Whitened regression data
#####

# For a LinearMixedModel with relative marginal covariance V = I + ZΛΛ'Z'
# (in the weighted space, if case weights are present), the GLS normal equations are
#   X'V⁻¹X β = X'V⁻¹y.
# The scores X'V⁻¹(y - Xβ̂) can be computed without forming V, because at the
# PLS solution V⁻¹(y - Xβ̂) = y - Xβ̂ - Zb̂, i.e. the conditional residuals.
# We therefore pass Vcov
#   - modelmatrix:         X (weighted, full-rank pivoted columns)
#   - crossmodelmatrix:    X'V⁻¹X = L_X L_X'  (L_X = feL(m))
#   - invcrossmodelmatrix: (X'V⁻¹X)⁻¹
#   - residuals:           V⁻¹(y - Xβ̂) (weighted conditional residuals)
# so that Vcov's sandwich B⁻¹ (Σ sᵢsᵢ') B⁻¹ uses the correct GLS bread and scores.
# The overall scale σ² cancels in the sandwich, so it is omitted.
# All quantities are in the pivoted, full-rank parameterization; see `_unpivot`.

"""
    Vcov.VcovData(m::LinearMixedModel)

Construct the `VcovData` used by Vcov.jl's sandwich estimators for the
fixed effects of `m`. The model matrix and bread are in the pivoted, full-rank
parameterization of `m`.
"""
function Vcov.VcovData(m::LinearMixedModel{T}) where {T}
    X = Matrix{Float64}(MixedModels.fullrankx(m))
    resid = Vector{Float64}(residuals(m))
    sqrtwts = m.sqrtwts
    if !isempty(sqrtwts)
        X .*= sqrtwts
        resid .*= sqrtwts
    end
    L = feL(m)
    Linv = inv(L)
    cm = Symmetric(Matrix{Float64}(L * L'))
    invcm = Symmetric(Matrix{Float64}(Linv' * Linv))
    return VcovData(X, cm, invcm, resid, nobs(m) - rank(m))
end

function Vcov.VcovData(::GeneralizedLinearMixedModel)
    throw(
        ArgumentError(
            "Vcov.jl covariance estimators are currently only " *
            "supported for LinearMixedModel",
        ),
    )
end

# Map a covariance matrix in the pivoted, full-rank parameterization back to the
# original coefficient order, filling in NaN for pivoted-out (rank-deficient) columns.
# This mirrors `vcov(::MixedModel)`.
function _unpivot(m::LinearMixedModel, permvcov::AbstractMatrix{T}) where {T}
    Xtrm = m.feterm
    iperm = invperm(Xtrm.piv)
    p = length(iperm)
    r = Xtrm.rank
    covmat = fill(zero(T) / zero(T), (p, p))
    copyto!(view(covmat, 1:r, 1:r), permvcov)
    return covmat[iperm, iperm]
end

#####
##### Estimators
#####

# Methods are defined separately for each CovarianceEstimator to avoid
# ambiguities with Vcov's methods for RegressionModel.

"""
    vcov(m::MixedModel, ::Vcov.SimpleCovariance)

Equivalent to `vcov(m)`: the model-based covariance matrix of the fixed effects.
"""
function StatsAPI.vcov(m::MixedModel, ::Vcov.SimpleCovariance)
    # validate model type consistently with the other estimators
    m isa LinearMixedModel || VcovData(m)
    return vcov(m)
end

end # module
