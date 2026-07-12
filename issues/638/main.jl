using CSV
using DataFrames
using Downloads
using GLM
using MixedModels

const CSV_URL = "https://github.com/JuliaStats/MixedModels.jl/files/9649005/data.csv"

data = CSV.read(Downloads.download(CSV_URL), DataFrame)

model_form = @formula(y ~ v1 + v2 + v3 + v4 + v5 +
                      (1 | pl3) + ((0 + v1) | pl3) +
                      (1 | pl5) + ((0 + v2) | pl5) +
                      ((0 + v3) | pl5) + ((0 + v4) | pl5) +
                      ((0 + v5) | pl5))

wts = data[!, :w]
contrasts = Dict(:pl3 => Grouping(), :pl5 => Grouping());
# contrasts = Dict(:pl3 => DummyCoding(), :pl5 => DummyCoding());
fit(MixedModel, model_form, data; wts, contrasts, amalgamate=false)

lm(@formula(y ~ v1 + v2 + v3 + v4 + v5), data; wts)
# y ~ 1 + v1 + v2 + v3 + v4 + v5

# Coefficients:
# ─────────────────────────────────────────────────────────────────────────────────────
#                     Coef.   Std. Error        t  Pr(>|t|)     Lower 95%     Upper 95%
# ─────────────────────────────────────────────────────────────────────────────────────
# (Intercept)  -0.000575762  0.000112393    -5.12    <1e-06  -0.000796048  -0.000355476
# v1           -0.934877     0.00206077   -453.65    <1e-99  -0.938916     -0.930838
# v2           -1.81368      0.00188045   -964.49    <1e-99  -1.81736      -1.80999
# v3            0.160488     0.000510854   314.16    <1e-99   0.159487      0.16149
# v4            1.5533       0.00112932   1375.43    <1e-99   1.55108       1.55551
# v5            1.16306      0.000691772  1681.28    <1e-99   1.16171       1.16442
# ─────────────────────────────────────────────────────────────────────────────────────

# R> summary(fm1)
# Linear mixed model fit by REML ['lmerMod']
# Formula: y ~ v1 + v2 + v3 + v4 + v5 + (1 | pl3) + ((0 + v1) | pl3) + (1 |
#     pl5) + ((0 + v2) | pl5) + ((0 + v3) | pl5) + ((0 + v4) |
#     pl5) + ((0 + v5) | pl5)
#    Data: data
# Weights: data$w

# REML criterion at convergence: 221644.3

# Scaled residuals:
#      Min       1Q   Median       3Q      Max
# -13.8621  -0.4886  -0.1377   0.1888  27.0177

# Random effects:
#  Groups   Name        Variance  Std.Dev.
#  pl5      v5          0.1602787 0.400348
#  pl5.1    v4          0.2347256 0.484485
#  pl5.2    v3          0.0473713 0.217649
#  pl5.3    v2          2.3506900 1.533196
#  pl5.4    (Intercept) 0.0000168 0.004099
#  pl3      v1          2.2690948 1.506351
#  pl3.1    (Intercept) 0.0000000 0.000000
#  Residual             2.5453766 1.595424
# Number of obs: 133841, groups:  pl5, 467; pl3, 79

# Fixed effects:
#               Estimate Std. Error t value
# (Intercept) -0.0007544  0.0008626  -0.875
# v1          -1.5365362  0.1839652  -8.352
# v2          -1.2907640  0.0927009 -13.924
# v3           0.2111352  0.0161907  13.041
# v4           0.9270981  0.0663387  13.975
# v5           0.4402297  0.0390687  11.268


# R> summary(refitML(fm1))
# Linear mixed model fit by maximum likelihood  ['lmerMod']
# Formula: y ~ v1 + v2 + v3 + v4 + v5 + (1 | pl3) + ((0 + v1) | pl3) + (1 |
#     pl5) + ((0 + v2) | pl5) + ((0 + v3) | pl5) + ((0 + v4) |
#     pl5) + ((0 + v5) | pl5)
#    Data: data
# Weights: data$w

#       AIC       BIC    logLik  deviance  df.resid
#  221640.9  221778.1 -110806.4  221612.9    133827

# Scaled residuals:
#      Min       1Q   Median       3Q      Max
# -13.8622  -0.4886  -0.1377   0.1888  27.0129

# Random effects:
#  Groups   Name        Variance  Std.Dev.
#  pl5      v5          1.615e-01 0.401829
#  pl5.1    v4          2.353e-01 0.485084
#  pl5.2    v3          4.693e-02 0.216635
#  pl5.3    v2          2.331e+00 1.526889
#  pl5.4    (Intercept) 1.651e-05 0.004064
#  pl3      v1          2.206e+00 1.485228
#  pl3.1    (Intercept) 0.000e+00 0.000000
#  Residual             2.545e+00 1.595419
# Number of obs: 133841, groups:  pl5, 467; pl3, 79

# Fixed effects:
#               Estimate Std. Error t value
# (Intercept) -0.0007564  0.0008610  -0.878
# v1          -1.5349996  0.1815460  -8.455
# v2          -1.2912605  0.0923754 -13.978
# v3           0.2111613  0.0161330  13.089
# v4           0.9269805  0.0664061  13.959
# v5           0.4399864  0.0391905  11.227

rtheta = [0.2515021687220257, 0.302059138995283, 0.1358219097194424, 0.9552822736385025, 0.0025389884728883316, 0.8849907215339659, 0.0]
r2jperm = [5, 4, 3, 2, 1, 7, 6]

fm1_unweighted = fit(MixedModel, model_form, data; contrasts)

fm1_weighted = LinearMixedModel(model_form, data; wts, contrasts)
# doesn't help
copy!(fm1_weighted.optsum.initial, fm1_unweighted.optsum.final)

fit!(fm1_weighted)

fm1 = fit(MixedModel, model_form, data; contrasts, wts)

# also doesn't help
updateL!(setθ!(fm1_weighted, rtheta[r2jperm]))

# nor does this work
slopes_form = @formula(y ~ 0 + v1 + v2 + v3 + v4 + v5 +
                      ((0 + v1) | pl3) + (1| pl5) +
                      ((0 + v2) | pl5) +
                      ((0 + v3) | pl5) + ((0 + v4) | pl5) +
                      ((0 + v5) | pl5))

fm2 = LinearMixedModel(slopes_form, data; wts, contrasts)

# but this does work
# fails with zero corr but otherwise gives similar estimates to lme
m_zc_pl3 = let f = @formula(y ~ v1 + v2 + v3 + v4 + v5 +
                     zerocorr(1 + v1 | pl3) +
                     (1 + v2 + v3 + v4 + v5 | pl5))

    fit(MixedModel, f, data; wts, contrasts)
end

m_no_int_pl3 = let f = @formula(y ~ v1 + v2 + v3 + v4 + v5 +
                     (0 + v1 | pl3) +
                     (1 + v2 + v3 + v4 + v5 | pl5))

    fit(MixedModel, f, data; wts, contrasts)
end

# let f = @formula(y ~ v1 + v2 + v3 + v4 + v5 +
#                          zerocorr(1 + v1 | pl3) +
#                          zerocorr(1 + v2 + v3 + v4 + v5 | pl5))
#     fit(MixedModel, f, data; wts, contrasts)
# end

using MixedModelsMakie
using CairoMakie
# ugh this covariance structure
splom!(Figure(), select(data, Not([:pl3, :pl5, :w, :y])))

select!(data, :,
        :pl3 => :pl3a,
        :pl3 => :pl3b,
        :pl5 => :pl5a,
        :pl5 => :pl5b,
        :pl5 => :pl5c,
        :pl5 => :pl5d,
        :pl5 => :pl5e)

contrasts = merge(contrasts, Dict(:pl3a => Grouping(),
                                  :pl3b => Grouping(),
                                  :pl5a => Grouping(),
                                  :pl5b => Grouping(),
                                  :pl5c => Grouping(),
                                  :pl5d => Grouping(),
                                  :pl5e => Grouping()))


using LinearAlgebra
MixedModels.rmulΛ!(A::Diagonal{T}, B::ReMat{T,1}) where {T} = rmul!(A, only(B.λ))
function MixedModels.rankUpdate!(C::Hermitian{T, Diagonal{T, Vector{T}}}, A::Diagonal{T, Vector{T}}, α, β) where {T}
    size(C) == size(A) || throw(DimensionMismatch("Diagonal matrices unequal size"))
    C.data.diag .*= β
    C.data.diag .+= α .* abs2.(A.diag)
    return C
end

m_form_split = let f = @formula(y ~ v1 + v2 + v3 + v4 + v5 +
                                (1 | pl3a) + ((0 + v1) | pl3b) +
                                (1 | pl5a) + ((0 + v2) | pl5b) +
                                ((0 + v3) | pl5c) + ((0 + v4) | pl5d) +
                                ((0 + v5) | pl5e))
    fit(MixedModel, f, data; wts, contrasts)
end

# test new kwarg

fit(MixedModel, model_form, data; wts, contrasts, amalgamate=false)
