lmb1 = LMMBase(Yield ~ 1|Batch, ds)

@test typeof(lmb1) == LMMBase
@test size(lmb1) == (30,1,6,1)
@test lmb1.Xs[1] == ones(1,30)
@test lmb1.facs[1].refs == rep(uint8([1:6]),1,5)
@test lmb1.X.m == ones((30,1))
@test size(lmb1.λ) == (1,)
@test all(map(istril,lmb1.λ))
@test lmb1.λ[1].UL == ones((1,1))
@test lmb1.Xty == [45825.0]
@test lmb1.fnms == {"Batch"}
@test isnested(lmb1)
@test MixedModels.isscalar(lmb1)
@test !isfit(lmb1)
@test grplevels(lmb1) == [6]
@test lower(lmb1) == [0.]
@test nobs(lmb1) == 30

lmb1.λ[1].UL[1,1] = 0.752583753954506
MixedModels.λZty!(lmb1)                 # prior to solving the system
@test_approx_eq lmb1.u[1] [5663.192748507658 5749.739880212426 5885.204955924238 5636.85231711925 6020.670031636048 5531.490591565619]

lmb1.β = [1527.5]
lmb1.u[1][:] = [-22.094892217084453 0.49099760482428484 35.84282515215955 -28.968858684621885 71.19465269949505 -56.46472455477185]
MixedModels.updateμ!(lmb1)

@test coef(lmb1) == [1527.5]
@test fixef(lmb1) == [1527.5]
@test_approx_eq MixedModels.rss(lmb1) 62668.1396425237
@test_approx_eq pwrss(lmb1) 73537.41156368206

zt = MixedModels.Zt(lmb1)
@test size(zt) == (6,30)
@test nnz(zt) == 30
@test all(zt.nzval .== 1.)

ztz = zt * zt'
@test size(ztz) == (6,6)
@test full(ztz) == 5.*eye(6)

zxt = MixedModels.ZXt(lmb1)
@test size(zxt) == (7,30)
@test nnz(zxt) == 60
@test all(zxt.nzval .== 1.)

lmb2 = LMMBase(Yield ~ 1|Batch, ds2)

@test typeof(lmb2) == LMMBase
@test size(lmb2) == (30,1,6,1)
@test lmb2.Xs[1] == ones(1,30)
@test lmb2.facs[1].refs == rep(uint8([1:6]),1,5)
@test lmb2.X.m == ones((30,1))
@test size(lmb2.λ) == (1,)
@test all(map(istril,lmb2.λ))
@test lmb2.λ[1].UL == ones((1,1))
@test_approx_eq lmb2.Xty [169.968]
@test lmb2.fnms == {"Batch"}
@test isnested(lmb2)
@test MixedModels.isscalar(lmb2)
@test !isfit(lmb2)
@test grplevels(lmb2) == [6]
@test lower(lmb2) == [0.]
@test nobs(lmb2) == 30

lmb2.λ[1].UL[1,1] = 0.0
MixedModels.λZty!(lmb2)                 # prior to solving the system
@test lmb2.u[1] == zeros(1,6)

lmb2.β = [5.6656]
MixedModels.updateμ!(lmb2)

@test coef(lmb2) == [5.6656]
@test fixef(lmb2) == [5.6656]
@test_approx_eq MixedModels.rss(lmb2) 400.3829792
@test_approx_eq pwrss(lmb2) 400.3829792

zt = MixedModels.Zt(lmb2)
@test size(zt) == (6,30)
@test nnz(zt) == 30
@test all(zt.nzval .== 1.)

ztz = zt * zt'
@test size(ztz) == (6,6)
@test full(ztz) == 5.*eye(6)

zxt = MixedModels.ZXt(lmb2)
@test size(zxt) == (7,30)
@test nnz(zxt) == 60
@test all(zxt.nzval .== 1.)

zxtzx = zxt * zxt'
@test vec(full(zxtzx[:,7])) == vcat(fill(5.,(6,)),30.)

lmb3 = LMMBase(Diameter ~ (1|Plate) + (1|Sample), pen)

@test typeof(lmb3) == LMMBase
@test size(lmb3) == (144,1,30,2)
@test lmb3.Xs == {ones(1,144),ones(1,144)}
@test lmb3.facs[1].refs == rep(uint8([1:24]),1,6)
@test lmb3.facs[2].refs == rep(uint8([1:6]),24,1)
@test lmb3.X.m == ones((144,1))
@test size(lmb3.λ) == (2,)
@test all(map(istril,lmb3.λ))
@test lmb3.λ[1].UL == ones((1,1))
@test lmb3.Xty == [3308.]
@test lmb3.fnms == {"Plate", "Sample"}
@test !isnested(lmb3)
@test MixedModels.isscalar(lmb3)
@test !isfit(lmb3)
@test grplevels(lmb3) == [24,6]
@test lower(lmb3) == [0.,0.]
@test nobs(lmb3) == 144

zt = MixedModels.Zt(lmb3)
@test size(zt) == (30,144)
@test nnz(zt) == 288
@test all(zt.nzval .== 1.)

ztz = zt * zt'
@test size(ztz) == (30,30)
@test issym(ztz)
@test full(ztz[1:24,1:24]) == 6.*eye(24)
@test full(ztz[25:30,25:30]) == 24.*eye(6)
@test full(ztz[1:24,25:30]) == ones(24,6)

zxt = MixedModels.ZXt(lmb3)
@test size(zxt) == (31,144)
@test nnz(zxt) == 288 + 144
@test all(zxt.nzval .== 1.)

zxtzx = zxt * zxt'
@test vec(full(zxtzx[:,31])) == vcat(fill(6.,(24,)),fill(24.,(6,)),144.)

lmb3.β[1] = mean(pen[:Diameter])
MixedModels.θ!(lmb3,[1.53759348675565, 3.21975344914413])
lmb3.u[1][:] = [0.523157593873041,   0.523157593873041,  0.11813235990679,  0.219388668398354, 
                0.0168760514152282, -0.286892874059457, -0.894430725008828, 0.523157593873039, 
               -0.489405491042583,  -0.489405491042581,  0.624413902364599, 0.320644976889914, 
                0.928182827839286,   0.320644976889915,  0.624413902364599, 0.01687605141523, 
               -0.185636565567894,  -0.185636565567894, -0.894430725008826, 0.624413902364599, 
               -0.590661799534141,  -0.185636565567895, -0.388149182551019,-0.793174416517266]
lmb3.u[2][:] = [0.678828300438623,  -0.313635860329661,    0.601493430768367,
               -0.0300746715387243, -0.00429638164863984, -0.932314817691712]
@test_approx_eq MixedModels.updateμ!(lmb) 35.24418682920359
@test_approx_eq pwrss(lmb) 43.549261374139476
