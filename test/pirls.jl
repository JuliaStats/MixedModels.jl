using DataFrames, Distributions, GLM, RCall, MixedModels

@rimport lme4

cbpp = rcopy(lme4.cbpp)
cbpp[:prop] = cbpp[:incidence] ./ cbpp[:size]
ttt = MixedModels.glmm(prop ~ 1 + period + (1|herd), cbpp, Binomial(), cbpp[:size], LogitLink());
