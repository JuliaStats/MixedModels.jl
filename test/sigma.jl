using RCall
using MixedModels
R"""
library(metafor)
dat <- escalc(measure="ZCOR", ri=ri, ni=ni, data=dat.molloy2014)
dat$study <- 1:nrow(dat)
"""
@rget dat

fit(MixedModel, @formula(yi ~ 1 + (1 | study)), dat;
    wts=1 ./ dat.vi,
    REML=true,
    contrasts=Dict(:study => Grouping()),
    σ=1)
