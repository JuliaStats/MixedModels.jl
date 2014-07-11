## ML fit to slp

lm3 = lmm(Reaction ~ Days + (Days|Subject), slp);

@test typeof(lm3) == LinearMixedModel{PLSOne}
@test size(lm3) == (180,2,36,1)
@test MixedModels.θ(lm3) == [1.,0.,1.]
@test lower(lm3) == [0.,-Inf,0.]

fit(lm3)

@test_approx_eq_eps deviance(lm3) 1751.9393445070384 1.e-6
@test_approx_eq_eps objective(lm3) 1751.9393445070384 1.e-6
@test_approx_eq coef(lm3) [251.40510484848477,10.4672859595959]
@test_approx_eq fixef(lm3) [251.40510484848477,10.4672859595959]
@test_approx_eq_eps stderr(lm3) [6.632246393963571,1.502190605041084] 1.e-3
@test_approx_eq_eps MixedModels.θ(lm3) [0.9292135718820399,0.01816527134999509,0.2226356241138231] 1.e-4
@test_approx_eq_eps std(lm3)[1] [23.78119476415131,5.717716838126193] 1.e-4
@test_approx_eq_eps scale(lm3) 25.59139819461323 1.e-6
@test_approx_eq_eps logdet(lm3) 8.390059690857333 1.e-5
@test_approx_eq_eps logdet(lm3,false) 73.90920980285422 1.e-4
@test_approx_eq diag(cor(lm3)[1]) ones(2)

#fit(reml!(lm3))                    # reml grad not yet written
                                        # fixed-effects estimates unchanged
#@test_approx_eq coef(lm3) [251.40510484848477,10.4672859595959]
#@test_approx_eq fixef(lm3) [251.40510484848477,10.4672859595959]
#@test_approx_eq stderr(lm3) [6.669402126263169,1.510606304414797]
#@test_approx_eq MixedModels.θ(lm3) [0.9292135717779286,0.018165271324834312,0.22263562408913865]
#@test isnan(deviance(lm3))
#@test_approx_eq objective(lm3) 1743.67380643908
#@test_approx_eq std(lm3)[1] [23.918164370001566,5.7295958427461064]
#@test_approx_eq std(lm3)[2] [25.735305686982493]
#@test_approx_eq triu(cholfact(lm3).UL) reshape([3.8957487178589947,0.0,2.3660528820280797,17.036408236726015],(2,2))
