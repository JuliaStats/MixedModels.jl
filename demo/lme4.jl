## Some of the examples from the lme4 package for R

using DataFrames, RDatasets, Distributions, MixedModels

ds = data("lme4", "Dyestuff")
fm1 = fit(LMMsimple(:(Yield ~ 1|Batch), ds), true);
show(fm1)

psts = data("lme4", "Pastes")
fm2 = fit(LMMsimple(:(strength ~ (1|sample) + (1|batch)), psts), true);
show(fm2)

pen = data("lme4", "Penicillin")
fm3 = fit(LMMsimple(:(diameter ~ (1|plate) + (1|sample)), pen), true);
show(fm3)

chem = data("mlmRev", "Chem97")
fm4 = fit(LMMsimple(:(score ~ (1|school) + (1|lea)), chem), true);
show(fm4)

fm5 = fit(LMMsimple(:(score ~ gcsecnt + (1|school) + (1|lea)), chem), true);
show(fm5)

fm6 = fit(reml(LMMsimple(:(score ~ gcsecnt + (1|school) + (1|lea)), chem)), true);
show(fm6)

@elapsed fm6 = fit(reml(LMMsimple(:(score ~ gcsecnt + (1|school) + (1|lea)), chem)))


