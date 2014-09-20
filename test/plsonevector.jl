## ML fit to slp

lmm₃ = lmm(Reaction ~ Days + (Days|Subject), slp);

@test typeof(lmm₃) == LinearMixedModel{PLSOne}
@test size(lmm₃) == (180,2,36,1)
@test MixedModels.θ(lmm₃) == [1.,0.,1.]
@test lower(lmm₃) == [0.,-Inf,0.]

fit(lmm₃)

@test_approx_eq_eps deviance(lmm₃) 1751.9393445 1.e-6
@test_approx_eq_eps objective(lmm₃) 1751.9393445 1.e-6
@test_approx_eq coef(lmm₃) [251.40510484848477,10.4672859595959]
@test_approx_eq fixef(lmm₃) [251.40510484848477,10.4672859595959]
@test_approx_eq_eps stderr(lmm₃) [6.632246393963571,1.502190605041084] 1.e-3
@test_approx_eq_eps MixedModels.θ(lmm₃) [0.9292135718820399,0.01816527134999509,0.2226356241138231] 1.e-4
@test_approx_eq_eps std(lmm₃)[1] [23.78058820861926,5.716840462986926] 1.e-4
@test_approx_eq_eps scale(lmm₃) 25.59181315643851 1.e-4
@test_approx_eq_eps logdet(lmm₃) 8.390384020157015 1.e-4
@test_approx_eq_eps logdet(lmm₃,false) 73.90337187545992 1.e-3
@test_approx_eq diag(cor(lmm₃)[1]) ones(2)

#fit(reml!(lmm₃))                    # reml grad not yet written
                                        # fixed-effects estimates unchanged
#@test_approx_eq coef(lmm₃) [251.40510484848477,10.4672859595959]
#@test_approx_eq fixef(lmm₃) [251.40510484848477,10.4672859595959]
#@test_approx_eq stderr(lmm₃) [6.669402126263169,1.510606304414797]
#@test_approx_eq MixedModels.θ(lmm₃) [0.9292135717779286,0.018165271324834312,0.22263562408913865]
#@test isnan(deviance(lmm₃))
#@test_approx_eq objective(lmm₃) 1743.67380643908
#@test_approx_eq std(lmm₃)[1] [23.918164370001566,5.7295958427461064]
#@test_approx_eq std(lmm₃)[2] [25.735305686982493]
#@test_approx_eq triu(cholfact(lmm₃).UL) reshape([3.8957487178589947,0.0,2.3660528820280797,17.036408236726015],(2,2))

lmm₄ = lmm(Reaction ~ Days + (1|Subject) + (0+Days|Subject), slp);

@test typeof(lmm₄) == LinearMixedModel{PLSOne}
@test size(lmm₄) == (180,2,36,1)
@test MixedModels.θ(lmm₄) == ones(2)
@test lower(lmm₄) == zeros(2)

fit(lmm₄)

@test_approx_eq_eps deviance(lmm₄) 1752.0032551398835 1.e-6
@test_approx_eq_eps objective(lmm₄) 1752.0032551398835 1.e-6
@test_approx_eq coef(lmm₄) [251.40510484848585,10.467285959595715]
@test_approx_eq fixef(lmm₄) [251.40510484848585,10.467285959595715]
@test_approx_eq_eps stderr(lmm₄) [6.7076736612161145,1.5193146383267642] 1.e-5
@test_approx_eq_eps MixedModels.θ(lmm₄) [0.9458106880922268,0.22692826607677266] 1.e-4
@test_approx_eq_eps std(lmm₄)[1] [24.171265054796056,5.799409265326568] 1.e-4
@test_approx_eq_eps scale(lmm₄) 25.55613439255097 1.e-6
@test_approx_eq_eps logdet(lmm₄) 8.358931021257781 1.e-5
@test_approx_eq_eps logdet(lmm₄,false) 74.46952585564611 1.e-4
@test_approx_eq diag(cor(lmm₄)[1]) ones(2)

tbl = MixedModels.lrt(lmm₄,lmm₃)

@test_approx_eq_eps tbl[:Deviance] [1752.0032551398835,1751.9393444636157] 1e-6
@test tbl[:Df] == [5,6]
