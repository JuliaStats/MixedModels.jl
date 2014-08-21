lmm₇ = lmm(Strength ~ (1|Sample) + (1|Batch), psts);

@test typeof(lmm₇) == LinearMixedModel{PLSDiag{Int32}}
@test size(lmm₇) == (60,1,40,2)
@test MixedModels.θ(lmm₇) == ones(2)
@test lower(lmm₇) == zeros(2)

fit(lmm₇)

@test_approx_eq_eps deviance(lmm₇) 247.99446586289676 1.e-6
@test_approx_eq_eps objective(lmm₇) 247.99446586289676 1.e-6
@test_approx_eq_eps coef(lmm₇) [60.05333333333329] 1.e-6
@test_approx_eq_eps fixef(lmm₇) [60.05333333333329] 1.e-6
@test_approx_eq_eps stderr(lmm₇) [0.6421358938945675] 1.e-8
@test_approx_eq_eps MixedModels.θ(lmm₇) [3.5268858714382905,1.3299230213750168] 1.e-5
@test_approx_eq_eps std(lmm₇)[1] [2.904069002535747] 1.e-6
@test_approx_eq_eps std(lmm₇)[2] [1.095070371687089] 1.e-6
@test_approx_eq_eps std(lmm₇)[3] [0.8234088395243269] 1.e-6
@test_approx_eq_eps scale(lmm₇) 0.8234088395243269 1.e-6
@test_approx_eq_eps logdet(lmm₇) 0.4973057812997239 1.e-5
@test_approx_eq_eps logdet(lmm₇,false) 101.0381339953986 1.e-4

fit(reml!(lmm₇))

@test isnan(deviance(lmm₇))
@test_approx_eq_eps objective(lmm₇) 246.99074585348615 1.e-6
@test_approx_eq_eps coef(lmm₇) [60.05333333333329] 1.e-6
@test_approx_eq_eps fixef(lmm₇) [60.05333333333329] 1.e-6
@test_approx_eq coef(lmm₇) [mean(lmm₇.y)] # balanced design
@test_approx_eq_eps stderr(lmm₇) [0.6768700853958879] 1.e-8
@test_approx_eq_eps MixedModels.θ(lmm₇) [3.52690188888386,1.5634603966070955] 1.e-5
@test_approx_eq_eps std(lmm₇)[1] [2.9040776178851986] 1.e-6
@test_approx_eq_eps std(lmm₇)[2] [1.28736508337448] 1.e-6
@test_approx_eq_eps std(lmm₇)[3] [0.8234066479821514] 1.e-6
@test_approx_eq_eps scale(lmm₇) 0.8234066479821514 1.e-6
@test_approx_eq_eps logdet(lmm₇) 0.3919438255955141 1.e-5
@test_approx_eq_eps logdet(lmm₇,false) 102.0919281931894 1.e-4
