inst = MixedModels.rdata("inst")

lmm₈ = lmm(Y ~ 1 + (1|S) + (1|D) + (1+Service|Dept), inst)

@test typeof(lmm₈) == LinearMixedModel{PLSGeneral{Int32}}
@test size(lmm₈) == (73421,1,4128,3)

fit(lmm₈)

@test_approx_eq_eps deviance(lmm₈) 237648.14982614366 1.e-3
@test_approx_eq_eps coef(lmm₈) [3.2684364013777296] 1.e-4

lmm₉ = lmm(Y ~ 1 + Service*Dept + (1|S) + (1|D), inst)

@test typeof(lmm₉) == LinearMixedModel{PLSDiag{Int32}}
@test size(lmm₉) == (73421,28,4100,2)

fit(lmm₉)

@test_approx_eq_eps deviance(lmm₉) 237585.55341516936 1.e-3
