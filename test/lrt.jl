lm6 = lmm(Reaction ~ Days + (1|Subject) + (0+Days|Subject), slp);  # should be converted to PLSOne

fit(lm6)

tt = MixedModels.lrt(lm6,lm3)

@test_approx_eq_eps tt[:Deviance] [1752.0032552746247,1751.9393450762673] 1e-6
@test all(tt[:Df] .== [5,6])
