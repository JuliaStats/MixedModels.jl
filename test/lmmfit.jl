## ML fit to ds

lm1 = lmm(Yield ~ 1 | Batch,ds);

@test typeof(lm1) == LinearMixedModel{PLSOne}
@test size(lm1) == (30,1,6,1)

fit(lm1)

@test_approx_eq_eps MixedModels.θ(lm1) [0.752580] 1.e-5
@test_approx_eq_eps deviance(lm1) 327.32705988 1.e-6
@test_approx_eq fixef(lm1) [1527.5]
@test_approx_eq coef(lm1) [1527.5]
@test_approx_eq_eps ranef(lm1)[1] [-16.62821559262611 0.369515902058292 26.974660850260033 -21.801438221443075 53.57980579846178 -42.494328736711275] 1.e-4
@test_approx_eq_eps ranef(lm1,true)[1] [-22.094942525296847 0.49099872278428663 35.842906763259194 -28.968924644278022 71.1948148037341 -56.46485312020319] 1.e-4
@test_approx_eq_eps std(lm1)[1] [37.26032326416065] 1.e-7
@test_approx_eq_eps std(lm1)[2] [49.510105062667854] 1.e-7
@test_approx_eq_eps logdet(lm1) 2.057840647724494 1.e-8
@test_approx_eq_eps logdet(lm1,false) 8.060140403625967 1.e-8
@test_approx_eq_eps scale(lm1) 49.510105062667854 1.e-7
@test_approx_eq_eps scale(lm1,true) 2451.2505033164093 1.e-3
@test_approx_eq_eps pwrss(lm1) 73537.51509949227 1.e-2
@test_approx_eq_eps stderr(lm1) [17.69454619561742] 1.e-7

## REML fit to ds

## fit(reml!(lm1))

## @test_approx_eq_eps std(lm1)[1] [42.00063130711604] 1.e-9
## @test_approx_eq_eps std(lm1)[2] [49.510093347813246] 1.e-9
## @test_approx_eq fixef(lm1) [1527.5]     # unchanged because of balanced design
## @test_approx_eq coef(lm1) [1527.5]
## @test_approx_eq_eps stderr(lm1) [19.383424615110936] 1.e-10
## @test_approx_eq objective(lm1) 319.6542768422625

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

lm5 = lmm(Diameter ~ (1|Plate) + (1|Sample), pen);

@test typeof(lm5) == LinearMixedModel{PLSDiag{Int32}}
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
