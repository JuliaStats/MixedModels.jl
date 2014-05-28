## ML fit to ds

lm1 = lmm(Yield ~ 1 | Batch,ds)

@test typeof(lm1) == LMMScalar1
@test size(lm1) == (30,1,6,1)
@test lm1.Ztnz == ones(30)
@test lm1.Ztrv == rep(uint8([1:6]),1,5)
@test lm1.Xt == ones((1,30))
@test length(lm1.theta) == 1
@test lm1.ZtZ == fill(5.,6)

fit(lm1)

@test_approx_eq theta(lm1) [0.752583753954506]
@test_approx_eq deviance(lm1) 327.3270598812219
@test_approx_eq fixef(lm1) [1527.5]
@test_approx_eq coef(lm1) [1527.5]
@test_approx_eq ranef(lm1) [-16.62825692795362 0.3695168206213314 26.974727905347223 -21.801492416650333 53.579938990073295 -42.49443437143737]
@test_approx_eq ranef(lm1,true) [-22.094892217084453 0.49099760482428484 35.84282515215955 -28.968858684621885 71.19465269949505 -56.46472455477185]
@test_approx_eq std(lm1) [37.260474496612346,49.51007020922851]
@test_approx_eq logdet(lm1) 2.057833608046211
@test_approx_eq logdet(lm1,false) 8.060182641695667
@test_approx_eq scale(lm1) 49.51007020922851
@test_approx_eq scale(lm1,true) 2451.247052122736
@test_approx_eq pwrss(lm1) 73537.41156368208
@test_approx_eq stderr(lm1) [17.694596021277448]

## REML fit to ds

fit(reml!(lm1))

@test_approx_eq std(lm1) [42.00063130711604,49.510093347813246]
@test_approx_eq fixef(lm1) [1527.5]     # unchanged because of balanced design
@test_approx_eq coef(lm1) [1527.5]
@test_approx_eq stderr(lm1) [19.383424615110936]
@test_approx_eq objective(lm1) 319.6542768422625

## ML fit to ds2

lm2 = fit(lmm(Yield ~ 1|Batch, ds))

@test_approx_eq deviance(lm2) 162.87303665382575
@test_approx_eq std(lm2) [0.0,3.653231351374652]
@test_approx_eq stderr(lm2) [0.6669857396443261]
@test_approx_eq coef(lm2) [5.6656]
@test_approx_eq logdet(lm2,false) 0.0
@test_approx_eq logdet(lm2) 3.4011973816621555

## ML fit to slp

lm3 = lmm(Reaction ~ Days + (Days|Subject), slp)

@test typeof(lm3) == LMMVector1
@test size(lm3) == (180,2,36,1)
@test theta(lm3) == [1.,0.,1.]
@test lower(lm3) == [0.,-Inf,0.]

fit(lm3)

@test_approx_eq deviance(lm3) 1751.9393445070389
@test_approx_eq objective(lm3) 1751.9393445070389
@test_approx_eq coef(lm3) [251.40510484848477,10.4672859595959]
@test_approx_eq fixef(lm3) [251.40510484848477,10.4672859595959]
@test_approx_eq stderr(lm3) [6.632246393560379,1.5021906049257874]
@test_approx_eq theta(lm3) [0.9292135717779286,0.01816527132483433,0.22263562408913878]
@test_approx_eq std(lm3) [23.78491450663324,5.69767584038342,25.591932394890165]
@test_approx_eq logdet(lm3) 8.390477202368283
@test_approx_eq logdet(lm3,false) 73.90169459565723
@test diag(cor(lm3)) == ones(2)
@test_approx_eq triu(cholfact(lm3).UL) reshape([3.895748717859,0.0,2.366052882028106,17.036408236726047],(2,2))

fit(reml!(lm3))
                                        # fixed-effects estimates unchanged
@test_approx_eq coef(lm3) [251.40510484848477,10.4672859595959]
@test_approx_eq fixef(lm3) [251.40510484848477,10.4672859595959]
@test_approx_eq stderr(lm3) [6.669402126263169,1.510606304414797]
@test_approx_eq theta(lm3) [0.9292135717779286,0.018165271324834312,0.22263562408913865]
@test isnan(deviance(lm3))
@test_approx_eq objective(lm3) 1743.67380643908
@test_approx_eq std(lm3) [23.918164370001566,5.7295958427461064,25.735305686982493]
@test_approx_eq triu(cholfact(lm3).UL) reshape([3.8957487178589947,0.0,2.3660528820280797,17.036408236726015],(2,2))
