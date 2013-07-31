## Some of the examples from the lme4 package for R

using RDatasets, MixedModels

ds = data("lme4", "Dyestuff")
fm1 = fit(lmer(:(Yield ~ 1|Batch), ds));
println(fm1)

ds2 = data("lme4", "Dyestuff2")
fm1a = fit(lmer(:(Yield ~ 1|Batch), ds2));
println(fm1a)

psts = data("lme4", "Pastes")
fm2 = fit(lmer(:(strength ~ (1|sample) + (1|batch)), psts));
println(fm2)

pen = data("lme4", "Penicillin")
fm3 = fit(lmer(:(diameter ~ (1|plate) + (1|sample)), pen));
println(fm3)

chem = data("mlmRev", "Chem97")
fm4 = fit(lmer(:(score ~ (1|school) + (1|lea)), chem));
println(fm4)

fm5 = fit(lmer(:(score ~ gcsecnt + (1|school) + (1|lea)), chem));
println(fm5)

@time fm6 = fit(reml!(lmer(:(score ~ gcsecnt + (1|school) + (1|lea)), chem)));
println(fm6)

inst = data("lme4", "InstEval")
@time fm7 = fit(lmer(:(y ~ dept*service + (1|s) + (1|d)), inst));
println(fm7)

sleep = data("lme4", "sleepstudy")
@time fm8 = fit(lmer(:(Reaction ~ Days + (Days | Subject)), sleep));
println(fm8)


