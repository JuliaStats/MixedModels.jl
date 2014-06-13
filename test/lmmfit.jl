## ML fit to ds

lm1 = lmm(Yield ~ 1 | Batch,ds);

@test typeof(lm1) == LinearMixedModel{PLSOne}
@test size(lm1) == (30,1,6,1)
@test lm1.Xs[1] == ones(1,30)
@test lm1.facs[1].refs == rep(uint8([1:6]),1,5)
@test lm1.X.m == ones((30,1))
@test size(lm1.λ) == (1,)
@test all(map(istril,lm1.λ))
@test lm1.λ[1].data == ones((1,1))
@test lm1.Xty == [45825.0]
@test lm1.fnms == {"Batch"}
@test isnested(lm1)
@test MixedModels.isscalar(lm1)
@test !isfit(lm1)
@test grplevels(lm1) == [6]
@test lower(lm1) == [0.]
@test nobs(lm1) == 30

Zt = MixedModels.zt(lm1)
@test size(Zt) == (6,30)
@test nnz(Zt) == 30
@test all(Zt.nzval .== 1.)

ztz = Zt * Zt'
@test size(ztz) == (6,6)
@test full(ztz) == 5.*eye(6)

ZXt = MixedModels.zxt(lm1)
@test size(ZXt) == (7,30)
@test nnz(ZXt) == 60
@test all(ZXt.nzval .== 1.)

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
@test_approx_eq MixedModels.updateμ!(lm1) 62668.1396425237

## REML fit to ds

fit(reml!(lm1))

@test_approx_eq_eps std(lm1)[1] [42.00063130711604] 1.e-9
@test_approx_eq_eps std(lm1)[2] [49.510093347813246] 1.e-9
@test_approx_eq fixef(lm1) [1527.5]     # unchanged because of balanced design
@test_approx_eq coef(lm1) [1527.5]
@test_approx_eq_eps stderr(lm1) [19.383424615110936] 1.e-10
@test_approx_eq objective(lm1) 319.6542768422625

## ML fit to ds2

lm2 = lmm(Yield ~ 1|Batch, ds2);

@test typeof(lm2) == LinearMixedModel{PLSOne}
@test size(lm2) == (30,1,6,1)
@test lm2.Xs[1] == ones(1,30)
@test lm2.facs[1].refs == rep(uint8([1:6]),1,5)
@test lm2.X.m == ones((30,1))
@test size(lm2.λ) == (1,)
@test all(map(istril,lm2.λ))
@test lm2.λ[1].data == ones((1,1))
@test_approx_eq lm2.Xty [169.968]
@test lm2.fnms == {"Batch"}
@test isnested(lm2)
@test MixedModels.isscalar(lm2)
@test !isfit(lm2)
@test grplevels(lm2) == [6]
@test lower(lm2) == [0.]
@test nobs(lm2) == 30

Zt = MixedModels.zt(lm2)
@test size(Zt) == (6,30)
@test nnz(Zt) == 30
@test all(Zt.nzval .== 1.)

ztz = Zt * Zt'
@test size(ztz) == (6,6)
@test full(ztz) == 5.*eye(6)

ZXt = MixedModels.zxt(lm2)
@test size(ZXt) == (7,30)
@test nnz(ZXt) == 60
@test all(ZXt.nzval .== 1.)

zxtzx = ZXt * ZXt'
@test vec(full(zxtzx[:,7])) == vcat(fill(5.,(6,)),30.)

fit(lm2)

@test_approx_eq deviance(lm2) 162.87303665382575
@test_approx_eq std(lm2)[1] [0.]
@test_approx_eq std(lm2)[2] [3.653231351374652]
@test_approx_eq stderr(lm2) [0.6669857396443261]
@test_approx_eq coef(lm2) [5.6656]
@test_approx_eq logdet(lm2,false) 0.0
@test_approx_eq logdet(lm2) 3.4011973816621555
@test_approx_eq coef(lm2) [5.6656]
@test_approx_eq fixef(lm2) [5.6656]
@test_approx_eq pwrss(lm2) 400.3829792

fit(reml!(lm2))

@test isnan(deviance(lm2))
@test_approx_eq objective(lm2) 161.82827781228846
@test_approx_eq std(lm2)[1] [0.]
@test_approx_eq std(lm2)[2] [3.715684274475726]
@test_approx_eq stderr(lm2) [0.678388031232524]
@test_approx_eq coef(lm2) [5.6656]
@test_approx_eq logdet(lm2,false) 0.0
@test_approx_eq logdet(lm2) 3.4011973816621555
@test_approx_eq fixef(lm2) [5.6656]
@test_approx_eq pwrss(lm2) 400.3829792

## ML fit to slp

lm3 = lmm(Reaction ~ Days + (Days|Subject), slp);

@test typeof(lm3) == LinearMixedModel{PLSOne}
@test size(lm3) == (180,2,36,1)
@test MixedModels.θ(lm3) == [1.,0.,1.]
@test lower(lm3) == [0.,-Inf,0.]

fit(lm3)

@test_approx_eq deviance(lm3) 1751.9393444889902
@test_approx_eq objective(lm3) 1751.9393444889902 
@test_approx_eq coef(lm3) [251.40510484848477,10.4672859595959]
@test_approx_eq fixef(lm3) [251.40510484848477,10.4672859595959]
@test_approx_eq stderr(lm3) [6.632122743884883,1.5022302138399877]
@test_approx_eq MixedModels.θ(lm3) [0.9291906056576971,0.0181657547591483,0.22264320562100107]
@test_approx_eq std(lm3)[1] [23.779759598309152,5.716798514016064]
@test_approx_eq scale(lm3) 25.591907035561803
@test_approx_eq logdet(lm3) 8.39045738514032
@test_approx_eq logdet(lm3,false) 73.90205130576032
@test_approx_eq diag(cor(lm3)[1]) ones(2)
@test_approx_eq vec(tril(lm3.s.Lt.UL)) [3.8958166968738537,2.365962697779489,0.0,17.03594216105133]

fit(reml!(lm3))
                                        # fixed-effects estimates unchanged
@test_approx_eq coef(lm3) [251.40510484848477,10.4672859595959]
@test_approx_eq fixef(lm3) [251.40510484848477,10.4672859595959]
@test_approx_eq stderr(lm3) [6.824911145217366,1.5464935701976237]
@test_approx_eq MixedModels.θ(lm3) [0.9668678597647252,0.014961558468032604,0.23106696899343698]
@test isnan(deviance(lm3))
@test_approx_eq objective(lm3) 1743.6282849227805
@test_approx_eq std(lm3)[1] [24.74266869079478,5.925510615250801]
@test_approx_eq std(lm3)[2] [25.590537983976]
#@test_approx_eq triu(cholfact(lm3).UL) reshape([3.8957487178589947,0.0,2.3660528820280797,17.036408236726015],(2,2))
