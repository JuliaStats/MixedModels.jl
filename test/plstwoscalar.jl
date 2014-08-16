lm5 = lmm(Diameter ~ (1|Plate) + (1|Sample), pen);

#@test typeof(lm5) == LinearMixedModel{PLSTwo}
@test size(lm5) == (144,1,30,2)
@test MixedModels.θ(lm5) == ones(2)
@test lower(lm5) == zeros(2)

fit(lm5)

@test_approx_eq_eps deviance(lm5) 332.1883489716297 1.e-6
@test_approx_eq_eps objective(lm5) 332.1883489716297 1.e-6
@test_approx_eq_eps coef(lm5) [22.972222222221742] 1.e-6
@test_approx_eq_eps fixef(lm5) [22.972222222221742] 1.e-6
@test_approx_eq coef(lm5)[1] mean(lm5.y) # balanced design
@test_approx_eq_eps stderr(lm5) [0.7444865834968265] 1.e-8
@test_approx_eq_eps MixedModels.θ(lm5) [1.5375793408948617,3.219226701391877] 1.e-5
@test_approx_eq_eps std(lm5)[1] [0.84557124623425] 1.e-6
@test_approx_eq_eps std(lm5)[2] [1.770370777889267] 1.e-6
@test_approx_eq_eps scale(lm5) 0.549936659361027 1.e-6
@test_approx_eq_eps logdet(lm5) -0.6057834486833463 1.e-5
@test_approx_eq_eps logdet(lm5,false) 95.74427699817353 1.e-4

fit(reml!(lm5))

@test isnan(deviance(lm5))
@test_approx_eq_eps objective(lm5) 330.86058900866897 1.e-6
@test_approx_eq_eps coef(lm5) [22.972222222221742] 1.e-6
@test_approx_eq_eps fixef(lm5) [22.972222222221742] 1.e-6
@test_approx_eq coef(lm5) [mean(lm5.y)] # balanced design
@test_approx_eq_eps stderr(lm5) [0.8086051290955466] 1.e-8
@test_approx_eq_eps MixedModels.θ(lm5) [1.5396785213580586,3.51256594128716] 1.e-5
@test_approx_eq_eps std(lm5)[1] [0.846703501327803] 1.e-6
@test_approx_eq_eps std(lm5)[2] [1.9316382217953854] 1.e-6
@test_approx_eq_eps scale(lm5) 0.5499222659682077 1.e-6
@test_approx_eq_eps logdet(lm5) -0.7710675334644331 1.e-5
@test_approx_eq_eps logdet(lm5,false) 96.83704281499782 1.e-4
