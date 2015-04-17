lmm₉ = lmm(Y ~ 1 + Service*Dept + (1|S) + (1|D), inst);

@test typeof(lmm₉) == LinearMixedModel{PLSDiag{Int32}}
@test size(lmm₉) == (73421,28,4100,2)

fit(lmm₉)

@test_approx_eq_eps deviance(lmm₉) 237585.55341516936 1.e-3
