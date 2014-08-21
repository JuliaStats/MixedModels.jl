## ML fit to ds

lmm₁ = lmm(Yield ~ 1 | Batch,ds);

@test typeof(lmm₁) == LinearMixedModel{PLSOne}
@test size(lmm₁) == (30,1,6,1)

fit(lmm₁)

@test_approx_eq_eps MixedModels.θ(lmm₁) [0.752580] 1.e-5
@test_approx_eq_eps deviance(lmm₁) 327.32705988 1.e-6
@test_approx_eq fixef(lmm₁) [1527.5]
@test_approx_eq coef(lmm₁) [1527.5]
@test_approx_eq_eps ranef(lmm₁)[1] [-16.62821559262611 0.369515902058292 26.974660850260033 -21.801438221443075 53.57980579846178 -42.494328736711275] 1.e-4
@test_approx_eq_eps ranef(lmm₁,true)[1] [-22.094942525296847 0.49099872278428663 35.842906763259194 -28.968924644278022 71.1948148037341 -56.46485312020319] 1.e-4
@test_approx_eq_eps std(lmm₁)[1] [37.26032326416065] 1.e-7
@test_approx_eq_eps std(lmm₁)[2] [49.510105062667854] 1.e-7
@test_approx_eq_eps logdet(lmm₁) 2.057840647724494 1.e-8
@test_approx_eq_eps logdet(lmm₁,false) 8.060140403625967 1.e-8
@test_approx_eq_eps scale(lmm₁) 49.510105062667854 1.e-7
@test_approx_eq_eps scale(lmm₁,true) 2451.2505033164093 1.e-3
@test_approx_eq_eps pwrss(lmm₁) 73537.51509949227 1.e-2
@test_approx_eq_eps stderr(lmm₁) [17.69454619561742] 1.e-7

## REML fit to ds

#fit(reml!(lmm₁))

##@test_approx_eq_eps std(lmm₁)[1] [37.8972980078109] 1.e-9
##@test_approx_eq_eps std(lmm₁)[2] [50.356492955140524] 1.e-9
##@test_approx_eq fixef(lmm₁) [1527.5]     # unchanged because of balanced design
##@test_approx_eq coef(lmm₁) [1527.5]
##@test_approx_eq_eps stderr(lmm₁) [19.383424615110936] 1.e-10
## @test_approx_eq objective(lmm₁) 319.6542768422625

## ML fit to ds2

lmm₂ = fit(lmm(Yield ~ 1|Batch, ds2))

@test_approx_eq deviance(lmm₂) 162.87303665382575
@test_approx_eq std(lmm₂)[1] [0.]
@test_approx_eq std(lmm₂)[2] [3.653231351374652]
@test_approx_eq stderr(lmm₂) [0.6669857396443261]
@test_approx_eq coef(lmm₂) [5.6656]
@test_approx_eq logdet(lmm₂,false) 0.0
@test_approx_eq logdet(lmm₂) 3.4011973816621555
