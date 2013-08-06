## Some of the examples from the lme4 package for R

using RDatasets, MixedModels

ds = data("lme4", "Dyestuff");
fm1 = lmm(:(Yield ~ 1|Batch), ds);
println(fm1)
typeof(fm1)

ds2 = data("lme4", "Dyestuff2");
@time fm1a = lmm(:(Yield ~ 1|Batch), ds2);
println(fm1a)
typeof(fm1a)

psts = data("lme4", "Pastes");
fm2 = lmm(:(strength ~ (1|sample) + (1|batch)), psts);
println(fm2)
typeof(fm2)

pen = data("lme4", "Penicillin");
@time fm3 = lmm(:(diameter ~ (1|plate) + (1|sample)), pen);
println(fm3)
typeof(fm3)

chem = data("mlmRev", "Chem97");
@time fm4 = lmm(:(score ~ (1|school) + (1|lea)), chem);
println(fm4)
typeof(fm4)

@time fm5 = lmm(:(score ~ gcsecnt + (1|school) + (1|lea)), chem);
println(fm5)
typeof(fm5)

@time fm6 = fit(reml!(lmm(Formula(:(score ~ gcsecnt + (1|school) + (1|lea))), chem; dofit=false)));
println(fm6)
typeof(fm6)

inst = data("lme4", "InstEval");
@time fm7 = lmm(:(y ~ dept*service + (1|s) + (1|d)), inst);
println(fm7)

sleep = data("lme4", "sleepstudy");
@time fm8 = lmm(:(Reaction ~ Days + (Days | Subject)), sleep);
println(fm8)
typeof(fm8)


