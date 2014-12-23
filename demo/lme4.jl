## Some of the examples from the lme4 package for R

using RDatasets, MixedModels

ds = dataset("lme4", "Dyestuff");
fm1 = fit(lmm(Yield ~ 1+(1|Batch), ds))
typeof(fm1)

ds2 = dataset("lme4", "Dyestuff2");
@time fm1a = fit(lmm(Yield ~ 1|Batch, ds2))
typeof(fm1a)

psts = dataset("lme4", "Pastes");
fm2 = fit(lmm(Strength ~ 1 + (1|Sample) + (1|Batch), psts))
typeof(fm2)

pen = dataset("lme4", "Penicillin");
@time fm3 = fit(lmm(Diameter ~ (1|Plate) + (1|Sample), pen))
typeof(fm3)

chem = dataset("mlmRev", "Chem97");
@time fm4 = fit(lmm(Score ~ (1|School) + (1|Lea), chem))
typeof(fm4)

@time fm5 = fit(lmm(Score ~ GCSECnt + (1|School) + (1|Lea), chem))
typeof(fm5)

@time fm6 = fit(reml!(lmm(Score ~ GCSECnt + (1|School) + (1|Lea), chem)))
typeof(fm6)

inst = dataset("lme4", "InstEval");
@time fm7 = fit(lmm(Y ~ Dept*Service + (1|S) + (1|D), inst))
typeof(fm7)

sleep = dataset("lme4", "sleepstudy");
@time fm8 = fit(lmm(Reaction ~ Days + (Days | Subject), sleep))
typeof(fm8)


