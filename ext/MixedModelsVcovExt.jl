module MixedModelsVcovExt

using GroupedArrays: GroupedArray
using LinearAlgebra: Cholesky, LowerTriangular, Symmetric, inv, rank
using MixedModels: MixedModels, LinearMixedModel, pivot
using MixedModels: feL, fnames
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
function Vcov.VcovData(m::LinearMixedModel)
    # VcovData fixes the eltype of the model matrix and residuals to Float64, and Vcov
    # accumulates the meat in Float64, so we work in Float64 throughout.
    # Matrix{Float64}(...) also copies: fullrankx(m) is a view into the model's storage
    # and is modified in place below.
    X = Matrix{Float64}(MixedModels.fullrankx(m))
    resid = Vector{Float64}(residuals(m))
    sqrtwts = m.sqrtwts
    if !isempty(sqrtwts)
        X .*= sqrtwts
        resid .*= sqrtwts
    end
    # if we refactor into an in-place multiplication, keep in mind that
    # feL(m) is a view into the model's blocked Cholesky factor
    # we need to guarantee the type and eltype
    L = LowerTriangular(convert(Matrix{Float64}, feL(m)))
    cm = Symmetric(L * L')
    # (LL')⁻¹ directly from the Cholesky factor (LAPACK potri)
    invcm = Symmetric(inv(Cholesky(L)))
    return VcovData(X, cm, invcm, resid, nobs(m) - rank(m))
end

# Map a covariance matrix in the pivoted, full-rank parameterization back to the
# original coefficient order, filling in NaN for pivoted-out (rank-deficient) columns.
# This mirrors `vcov(::MixedModel)`.
function _unpivot(m::LinearMixedModel, permvcov::AbstractMatrix{T}) where {T}
    iperm = invperm(pivot(m))
    p = length(iperm)
    r = rank(m)
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
    vcov(m::LinearMixedModel, ::Vcov.SimpleCovariance)

Equivalent to `vcov(m)`: the model-based covariance matrix of the fixed effects.
"""
function StatsAPI.vcov(m::LinearMixedModel, ::Vcov.SimpleCovariance)
    return vcov(m)
end

"""
    vcov(m::LinearMixedModel, ::Vcov.RobustCovariance)

Heteroskedasticity-robust (HC1) sandwich estimator of the covariance of the fixed effects,
```math
(X'V^{-1}X)^{-1} \\left(\\frac{n}{n-p}\\sum_i s_i s_i'\\right) (X'V^{-1}X)^{-1},
```
where ``s_i = x_i [V^{-1}(y - X\\hat\\beta)]_i`` is the contribution of observation ``i``
to the score of the fixed effects.

!!! warning
    This estimator treats the per-observation score contributions as independent.
    In a mixed model, contributions from observations sharing a level of a grouping
    factor are correlated, so `Vcov.cluster` over the grouping factor(s) is usually
    more appropriate.

Only HC1 is supported: the HC2/HC3 leverage corrections are not defined for
mixed models here and throw an `ArgumentError`.
"""
function StatsAPI.vcov(m::LinearMixedModel, c::Vcov.RobustCovariance)
    correction = c.correction
    if correction != :hc1
        throw(
            ArgumentError(
                "only HC1 (`Vcov.robust()`/`Vcov.hc(1)`) is supported " *
                "for mixed models, got $(uppercase(string(correction)))",
            ),
        )
    end
    return _unpivot(m, vcov(VcovData(m), c))
end

"""
    vcov(m::LinearMixedModel, c::Vcov.ClusterCovariance)

Cluster-robust (CR1) sandwich estimator of the covariance of the fixed effects,
```math
(X'V^{-1}X)^{-1} \\left(\\frac{n-1}{n-p}\\frac{G}{G-1}\\sum_g s_g s_g'\\right) (X'V^{-1}X)^{-1},
```
where ``s_g`` is the sum of the score contributions ``x_i [V^{-1}(y - X\\hat\\beta)]_i``
of the observations in cluster ``g`` and ``G`` is the number of clusters.
Multiway clustering follows Cameron, Gelbach & Miller (2011).

Clusters given by name, e.g. `Vcov.cluster(:subj)`, must be grouping factors of `m`.
To cluster on other variables, pass already-materialized clusters with one entry per
observation used in fitting `m`, e.g.
`Vcov.cluster((:school,), (GroupedArray(school),))`.

When the clusters are the only grouping factor of `m` (or `m`'s grouping factors are
nested within them), this is the Liang-Zeger sandwich estimator.
"""
function StatsAPI.vcov(m::LinearMixedModel, c::Vcov.ClusterCovariance)
    vd = VcovData(m)
    return _unpivot(m, vcov(vd, _materialize(m, c)))
end

function _materialize(m::LinearMixedModel, c::Vcov.ClusterCovariance)
    if !isnothing(c.clusters)
        for (n, g) in zip(Vcov.names(c), c.clusters)
            length(g) == nobs(m) ||
                throw(
                    DimensionMismatch(
                        "cluster $(n) has length $(length(g)), " *
                        "but the model has $(nobs(m)) observations",
                    ),
                )
        end
        return c
    end
    fns = fnames(m)
    clusters = map(Vcov.names(c)) do n
        i = findfirst(==(n), fns)
        isnothing(i) &&
            throw(
                ArgumentError(
                    "$(n) is not a grouping factor of the model " *
                    "(grouping factors: $(join(fns, ", "))); " *
                    "pass materialized clusters to cluster on other variables",
                ),
            )
        return GroupedArray(m.reterms[i].refs)
    end
    return Vcov.cluster(Vcov.names(c), clusters)
end

end # module
