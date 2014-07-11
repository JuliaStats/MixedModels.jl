lm4 = lmm(Strength ~ (1|Sample) + (1|Batch), psts);

@test typeof(lm4) == LinearMixedModel{PLSDiag{Int32}}
@test size(lm4) == (60,1,40,2)
@test MixedModels.θ(lm4) == ones(2)
@test lower(lm4) == zeros(2)

fit(lm4)

@test_approx_eq_eps deviance(lm4) 247.99446595133628 1.e-6
@test_approx_eq_eps objective(lm4) 247.99446595133628 1.e-6
@test_approx_eq_eps coef(lm4) [60.05333333333329] 1.e-6
@test_approx_eq_eps fixef(lm4) [60.05333333333329] 1.e-6
@test_approx_eq_eps stderr(lm4) [0.6421028764099568] 1.e-8
@test_approx_eq_eps MixedModels.θ(lm4) [3.526777369707641,1.3296826663749939] 1.e-5
@test_approx_eq_eps std(lm4)[1] [2.9040438837658265] 1.e-6
@test_approx_eq_eps std(lm4)[2] [1.0948966747384568] 1.e-6
@test_approx_eq_eps std(lm4)[3] [0.8234270494954896] 1.e-6
@test_approx_eq_eps scale(lm4) 0.8234270494954896 1.e-6
@test_approx_eq_eps logdet(lm4) 0.4974528505858654 1.e-5
@test_approx_eq_eps logdet(lm4,false) 101.03548027169548 1.e-4

fit(reml!(lm4))

@test isnan(deviance(lm4))
@test_approx_eq_eps objective(lm4) 246.9907458682234 1.e-6
@test_approx_eq_eps coef(lm4) [60.05333333333329] 1.e-6
@test_approx_eq_eps fixef(lm4) [60.05333333333329] 1.e-6
@test_approx_eq coef(lm4) [mean(lm4.y)] # balanced design
@test_approx_eq_eps stderr(lm4) [0.6768868401974097] 1.e-8
@test_approx_eq_eps MixedModels.θ(lm4) [3.5268709921936656,1.563595307807885] 1.e-5
@test_approx_eq_eps std(lm4)[1] [2.9040490215476704] 1.e-6
@test_approx_eq_eps std(lm4)[2] [1.2874747712027106] 1.e-6
@test_approx_eq_eps std(lm4)[3] [0.8234066479821514] 1.e-6
@test_approx_eq_eps scale(lm4) 0.8234066479821514 1.e-6
@test_approx_eq_eps logdet(lm4) 0.39189214615273066 1.e-5
@test_approx_eq_eps logdet(lm4,false) 102.09210811544975 1.e-4
