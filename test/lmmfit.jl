## ML fit to ds

lm1 = lmm(Yield ~ 1 | Batch,ds);

@test typeof(lm1) == LinearMixedModel
@test size(lm1) == (30,1,6,1)

fit(lm1)

@test_approx_eq_eps MixedModels.θ(lm1) [0.752583753954506] 1.e-10
@test_approx_eq deviance(lm1) 327.3270598812219
@test_approx_eq fixef(lm1) [1527.5]
@test_approx_eq coef(lm1) [1527.5]
@test_approx_eq ranef(lm1)[1] [-16.62825692795362 0.3695168206213314 26.974727905347223 -21.801492416650333 53.579938990073295 -42.49443437143737]
@test_approx_eq ranef(lm1,true)[1] [-22.094892217084453 0.49099760482428484 35.84282515215955 -28.968858684621885 71.19465269949505 -56.46472455477185]
@test_approx_eq_eps std(lm1)[1] [37.260474496612346] 1.e-9
@test_approx_eq_eps std(lm1)[2] [49.51007020922851] 1.e-9
@test_approx_eq_eps logdet(lm1) 2.057833608046211 1.e-10
@test_approx_eq_eps logdet(lm1,false) 8.060182641695667 1.e-10
@test_approx_eq scale(lm1) 49.51007020922851
@test_approx_eq_eps scale(lm1,true) 2451.247052122736 1.e-8
@test_approx_eq_eps pwrss(lm1) 73537.41156368208 1.e-6
@test_approx_eq_eps stderr(lm1) [17.694596021277448] 1.e-10

## REML fit to ds

fit(reml!(lm1))

@test_approx_eq_eps std(lm1)[1] [42.00063130711604] 1.e-9
@test_approx_eq_eps std(lm1)[2] [49.510093347813246] 1.e-9
@test_approx_eq fixef(lm1) [1527.5]     # unchanged because of balanced design
@test_approx_eq coef(lm1) [1527.5]
@test_approx_eq_eps stderr(lm1) [19.383424615110936] 1.e-10
@test_approx_eq objective(lm1) 319.6542768422625

## ML fit to ds2

lm2 = fit(lmm(Yield ~ 1|Batch, ds2))

@test_approx_eq deviance(lm2) 162.87303665382575
@test_approx_eq std(lm2)[1] [0.]
@test_approx_eq std(lm2)[2] [3.653231351374652]
@test_approx_eq stderr(lm2) [0.6669857396443261]
@test_approx_eq coef(lm2) [5.6656]
@test_approx_eq logdet(lm2,false) 0.0
@test_approx_eq logdet(lm2) 3.4011973816621555

## ML fit to slp

lm3 = lmm(Reaction ~ Days + (Days|Subject), slp);

@test typeof(lm3) == LinearMixedModel
@test typeof(lm3.pls) == DeltaLeaf
@test size(lm3) == (180,2,36,1)
@test MixedModels.θ(lm3) == [1.,0.,1.]
@test lower(lm3) == [0.,-Inf,0.]

fit(lm3)

@test_approx_eq deviance(lm3) 1751.9393444889902
@test_approx_eq objective(lm3) 1751.9393444889902 
@test_approx_eq coef(lm3) [251.40510484848477,10.4672859595959]
@test_approx_eq fixef(lm3) [251.40510484848477,10.4672859595959]
@test_approx_eq stderr(lm3) [6.632122744453551,1.5022302138615442]
@test_approx_eq MixedModels.θ(lm3) [0.9291906057869259,0.018165754769515766,0.22264320562793902]
@test_approx_eq std(lm3)[1] [23.779759601293264,5.716798514136933]
@test_approx_eq scale(lm3) 25.59190703521408
@test_approx_eq logdet(lm3) 8.390457384868602
@test_approx_eq logdet(lm3,false) 73.90205131065146
@test diag(cor(lm3)[1]) == ones(2)
@test_approx_eq tril(lm3.pls.Lt.UL) reshape([3.89581669645341,2.365962696639337,0.0,17.035942160575402],(2,2))

fit(reml!(lm3))
                                        # fixed-effects estimates unchanged
@test_approx_eq coef(lm3) [251.40510484848477,10.4672859595959]
@test_approx_eq fixef(lm3) [251.40510484848477,10.4672859595959]
#@test_approx_eq stderr(lm3) [6.669402126263169,1.510606304414797]
#@test_approx_eq MixedModels.θ(lm3) [0.9292135717779286,0.018165271324834312,0.22263562408913865]
@test isnan(deviance(lm3))
#@test_approx_eq objective(lm3) 1743.67380643908
#@test_approx_eq std(lm3)[1] [23.918164370001566,5.7295958427461064]
#@test_approx_eq std(lm3)[2] [25.735305686982493]
#@test_approx_eq triu(cholfact(lm3).UL) reshape([3.8957487178589947,0.0,2.3660528820280797,17.036408236726015],(2,2))
