## Some of the examples from the lme4 package for R

using RDatasets, MixedModels

ds = data("lme4", "Dyestuff")
fm1 = LMMsimple(:(Yield ~ 1|Batch), ds)

ds2 = data("lme4", "Dyestuff2")
fm1a = LMMsimple(:(Yield ~ 1|Batch), ds2)

psts = data("lme4", "Pastes")
fm2 = LMMsimple(:(strength ~ (1|sample) + (1|batch)), psts)

pen = data("lme4", "Penicillin")
fm3 = LMMsimple(:(diameter ~ (1|plate) + (1|sample)), pen)

chem = data("mlmRev", "Chem97")
fm4 = LMMsimple(:(score ~ (1|school) + (1|lea)), chem)

fm5 = LMMsimple(:(score ~ gcsecnt + (1|school) + (1|lea)), chem)

fm6 = reml(LMMsimple(:(score ~ gcsecnt + (1|school) + (1|lea)), chem))

@elapsed fm6 = fit(reml(LMMsimple(:(score ~ gcsecnt + (1|school) + (1|lea)), chem)))


