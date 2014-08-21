lmm₅ = lmm(Diameter ~ (1|Plate) + (1|Sample), pen);

@test typeof(lmm₅) == LinearMixedModel{PLSTwo}
@test size(lmm₅) == (144,1,30,2)
@test MixedModels.θ(lmm₅) == ones(2)
@test lower(lmm₅) == zeros(2)

fit(lmm₅)

@test_approx_eq_eps deviance(lmm₅) 332.18834867227616 1.e-6
@test_approx_eq_eps objective(lmm₅) 332.18834867227616 1.e-6
@test_approx_eq_eps coef(lmm₅) [22.972222222220246] 1.e-6
@test_approx_eq_eps fixef(lmm₅) [22.972222222220246] 1.e-6
@test_approx_eq coef(lmm₅)[1] mean(lmm₅.y) # balanced design
@test_approx_eq_eps stderr(lmm₅) [0.7445959664728862] 1.e-8
@test_approx_eq_eps MixedModels.θ(lmm₅) [1.5375772508259862,3.2197509968696556] 1.e-5
@test_approx_eq_eps std(lmm₅)[1] [0.845564604484871] 1.e-6
@test_approx_eq_eps std(lmm₅)[2] [1.770647605995295] 1.e-6
@test_approx_eq_eps scale(lmm₅) 0.5499330872843194 1.e-6
@test_approx_eq_eps logdet(lmm₅) -0.6060902660855345 1.e-5
@test_approx_eq_eps logdet(lmm₅,false) 95.7461473895903 1.e-4

fit(reml!(lmm₅))

@test isnan(deviance(lmm₅))
@test_approx_eq_eps objective(lmm₅) 330.86058899122156 1.e-6
@test_approx_eq_eps coef(lmm₅) [22.972222222222218] 1.e-6
@test_approx_eq_eps fixef(lmm₅) [22.972222222222218] 1.e-6
@test_approx_eq coef(lmm₅) [mean(lmm₅.y)] # balanced design
@test_approx_eq_eps stderr(lmm₅) [0.8085723219702367] 1.e-8
@test_approx_eq_eps MixedModels.θ(lmm₅) [1.5396804010953746,3.5124108917799424] 1.e-5
@test_approx_eq_eps std(lmm₅)[1] [0.8467056820846379] 1.e-6
@test_approx_eq_eps std(lmm₅)[2] [1.9315555733321477] 1.e-6
@test_approx_eq_eps scale(lmm₅) 0.5499230109588108 1.e-6
@test_approx_eq_eps logdet(lmm₅) -0.7709836773959788 1.e-5
@test_approx_eq_eps logdet(lmm₅,false) 96.83657149187079 1.e-4
