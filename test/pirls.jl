using Distributions,RDatasets,MixedModels

cbpp = dataset("lme4","cbpp");
cbpp[:sz] = convert(Vector{Float64},cbpp[:Size]);
cbpp[:prop] = array(cbpp[:Incidence]) ./ array(cbpp[:sz]);
ttt = glmm(prop ~ 1 + Period + (1|Herd), cbpp, Binomial(), array(cbpp[:sz]))
