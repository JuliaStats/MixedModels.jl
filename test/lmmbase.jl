lmb1 = LMMBase(Yield ~ 1|Batch, ds)

@test typeof(lmb1) == LMMBase
@test size(lmb1) == (30,1,6,1)
@test lmb1.Xs[1] == ones(1,30)
@test lmb1.facs[1].refs == rep(uint8([1:6]),1,5)
@test lmb1.X.m == ones((30,1))
@test size(lmb1.λ) == (1,)
@test all(map(istril,lmb1.λ))
@test lmb1.λ[1].data == ones((1,1))
@test lmb1.Xty == [45825.0]
@test lmb1.fnms == {"Batch"}
@test isnested(lmb1)
@test MixedModels.isscalar(lmb1)
@test !isfit(lmb1)
@test grplevels(lmb1) == [6]
@test lower(lmb1) == [0.]
@test nobs(lmb1) == 30

MixedModels.θ!(lmb1,[0.752583753954506])
MixedModels.λtZty!(lmb1)                 # prior to solving the system
@test_approx_eq lmb1.u[1] [5663.192748507658 5749.739880212426 5885.204955924238 5636.85231711925 6020.670031636048 5531.490591565619]

lmb1.β = [1527.5]
lmb1.u[1][:] = [-22.094892217084453 0.49099760482428484 35.84282515215955 -28.968858684621885 71.19465269949505 -56.46472455477185]

@test coef(lmb1) == [1527.5]
@test fixef(lmb1) == [1527.5]
@test_approx_eq MixedModels.updateμ!(lmb1) 62668.1396425237
@test_approx_eq pwrss(lmb1) 73537.41156368206
@test_approx_eq std(lmb1)[1] [37.260474496612346]
@test_approx_eq scale(lmb1) 49.5100702092285

Zt = MixedModels.zt(lmb1)
@test size(Zt) == (6,30)
@test nnz(Zt) == 30
@test all(Zt.nzval .== 1.)

ztz = Zt * Zt'
@test size(ztz) == (6,6)
@test full(ztz) == 5.*eye(6)

ZXt = MixedModels.zxt(lmb1)
@test size(ZXt) == (7,30)
@test nnz(ZXt) == 60
@test all(ZXt.nzval .== 1.)

lmb2 = LMMBase(Yield ~ 1|Batch, ds2)

@test typeof(lmb2) == LMMBase
@test size(lmb2) == (30,1,6,1)
@test lmb2.Xs[1] == ones(1,30)
@test lmb2.facs[1].refs == rep(uint8([1:6]),1,5)
@test lmb2.X.m == ones((30,1))
@test size(lmb2.λ) == (1,)
@test all(map(istril,lmb2.λ))
@test lmb2.λ[1].data == ones((1,1))
@test_approx_eq lmb2.Xty [169.968]
@test lmb2.fnms == {"Batch"}
@test isnested(lmb2)
@test MixedModels.isscalar(lmb2)
@test !isfit(lmb2)
@test grplevels(lmb2) == [6]
@test lower(lmb2) == [0.]
@test nobs(lmb2) == 30

MixedModels.θ!(lmb2,[0.])
MixedModels.λtZty!(lmb2)                 # prior to solving the system
@test lmb2.u[1] == zeros(1,6)

lmb2.β = [5.6656]
@test_approx_eq MixedModels.updateμ!(lmb2) 400.3829792

@test coef(lmb2) == [5.6656]
@test fixef(lmb2) == [5.6656]
@test_approx_eq pwrss(lmb2) 400.3829792

Zt = MixedModels.zt(lmb2)
@test size(Zt) == (6,30)
@test nnz(Zt) == 30
@test all(Zt.nzval .== 1.)

ztz = Zt * Zt'
@test size(ztz) == (6,6)
@test full(ztz) == 5.*eye(6)

ZXt = MixedModels.zxt(lmb2)
@test size(ZXt) == (7,30)
@test nnz(ZXt) == 60
@test all(ZXt.nzval .== 1.)

zxtzx = ZXt * ZXt'
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
@test lmb3.λ[1].data == ones((1,1))
@test lmb3.λ[2].data == ones((1,1))
@test MixedModels.θ(lmb3) == ones(2)
@test lmb3.Xty == [3308.]
@test lmb3.fnms == {"Plate", "Sample"}
@test !isnested(lmb3)
@test MixedModels.isscalar(lmb3)
@test !isfit(lmb3)
@test grplevels(lmb3) == [24,6]
@test lower(lmb3) == [0.,0.]
@test nobs(lmb3) == 144

Zt = MixedModels.zt(lmb3)
@test size(Zt) == (30,144)
@test nnz(Zt) == 288
@test all(Zt.nzval .== 1.)

ztz = Zt * Zt'
@test size(ztz) == (30,30)
@test issym(ztz)
@test full(ztz[1:24,1:24]) == 6.*eye(24)
@test full(ztz[25:30,25:30]) == 24.*eye(6)
@test full(ztz[1:24,25:30]) == ones(24,6)

ZXt = MixedModels.zxt(lmb3)
@test size(ZXt) == (31,144)
@test nnz(ZXt) == 288 + 144
@test all(ZXt.nzval .== 1.)

zxtzx = ZXt * ZXt'
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
@test_approx_eq MixedModels.updateμ!(lmb3) 35.24418682920359
@test_approx_eq pwrss(lmb3) 43.549261374139476

lmb4 = LMMBase(Strength ~ (1|Sample) + (1|Batch), psts)

@test size(lmb4) == (60,1,40,2)
@test lmb4.Xs == {ones(1,60),ones(1,60)}
@test lmb4.facs[1].refs == rep(uint8([1:30]),1,2)
@test lmb4.facs[2].refs == rep(uint8([1:10]),1,6)
@test lmb4.X.m == ones((60,1))
@test size(lmb4.λ) == (2,)
@test all(map(istril,lmb4.λ))
@test lmb4.λ[1].data == ones((1,1))
@test lmb4.λ[2].data == ones((1,1))
@test MixedModels.θ(lmb4) == ones(2)
@test lmb4.Xty == [3603.2]
@test lmb4.fnms == {"Sample","Batch"}
@test isnested(lmb4)
@test MixedModels.isscalar(lmb4)
@test !isfit(lmb4)
@test grplevels(lmb4) == [30,10]
@test lower(lmb4) == [0.,0.]
@test nobs(lmb4) == 60

Zt = MixedModels.zt(lmb4)
@test size(Zt) == (40,60)
@test nnz(Zt) == 120
@test all(Zt.nzval .== 1.)

ztz = Zt * Zt'
@test size(ztz) == (40,40)
@test issym(ztz)
#@test ztz[1:30,1:30] == 2.*speye(30)  ## temporarily disabled b/c of indexing problems 
#@test ztz[31:40,31:40] == 6.*speye(10)

ZXt = MixedModels.zxt(lmb4)
@test size(ZXt) == (41,60)
@test nnz(ZXt) == 180
@test all(ZXt.nzval .== 1.)

MixedModels.θ!(lmb4,[3.52690173547125, 1.32991216014294])
lmb4.β[:] = mean(psts[:Strength])
lmb4.u[1][:] = [0.545971198951695, 0.137103593679235,   0.60048687965469,   0.235984796077345,
               -0.718039616225064, 0.0451799136168632, -0.690714090621525,  0.781209288359335,
                1.06741661205006, -0.872408266906441,  -0.599829863391465,  1.26733220068611,
               -1.02093312625967, -1.1981090885444,    -0.189568995538997,  1.03897735543894,
               -0.446574943717672,-0.0240784182694611,  0.692692288930594, -0.0432694005598349,
               -0.738344329523022,-0.448175657335874,   1.26906828480846,   0.95560312076624,
               -1.32334404896573,  1.18437726337204,   -0.655526960354038, -0.225103342630958,
               -0.116071981224968,-0.51131066632168]
lmb4.u[2][:] = [0.484001087724721, -0.164735309978546,  0.436621463173287, -0.0772652338837034,
               -0.908230956784477,  0.214301686432351, -0.0335301958362921, 0.669874999426066,
               -0.299585010624761, -0.321452529648366]
@test_approx_eq MixedModels.updateμ!(lmb4) 21.049799756610106
@test_approx_eq pwrss(lmb4) 40.68000037028114

lmb5 = LMMBase(Reaction ~ Days + (Days|Subject), slp)

@test size(lmb5) == (180,2,36,1)
XX = hcat(ones(180),rep([0.:9.],18,1))
@test lmb5.Xs == {XX'}
@test lmb5.facs[1].refs == rep(uint8([1:18]),1,10)
@test lmb5.X.m == XX
@test size(lmb5.λ) == (1,)
@test all(map(istril,lmb5.λ))
@test full(lmb5.λ[1]) == eye(2)
@test MixedModels.θ(lmb5) == [1.,0.,1.]
@test_approx_eq lmb5.Xty [53731.4205,257335.3119]
@test lmb5.fnms == {"Subject"}
@test isnested(lmb5)
@test !MixedModels.isscalar(lmb5)
@test !isfit(lmb5)
@test grplevels(lmb5) == [18]
@test lower(lmb5) == [0.,-Inf,0.]
@test nobs(lmb5) == 180

Zt = MixedModels.zt(lmb5)
@test size(Zt) == (36,180)
@test nnz(Zt) == 360
@test issubset(Zt.nzval,[0.:9.])

ztz = Zt * Zt'
@test size(ztz) == (36,36)
@test issym(ztz)
evens = 2:2:36
odds = 1:2:35
speye18 = speye(18)
#@test ztz[evens,evens] == 285.*speye18
#@test ztz[odds,odds] == 10.*speye18
#@test ztz[evens,odds] == ztz[odds,evens] == 45.*speye18

ZXt = MixedModels.zxt(lmb5)
@test size(ZXt) == (38,180)
@test nnz(ZXt) == 702
@test issubset(ZXt.nzval,[0.:9.])

zxtzx = ZXt * ZXt'
@test size(zxtzx) == (38,38)
@test nnz(zxtzx) == 220
@test countmap(zxtzx.nzval) == [810.0=>2,10.0=>54,285.0=>54,180.0=>1,5130.0=>1,45.0=>108]

MixedModels.θ!(lmb5,[0.929225333147176, 0.0181656125028504, 0.222645384404873])
MixedModels.λtZty!(lmb5)
@test_approx_eq lmb5.λ[1]*lmb5.Zty[1] lmb5.u[1]
lmb5.β[:] = [251.405104848484, 10.4672859595955]
lmb5.u[1][:] = [  3.03013868860485, 40.5150534730329, -43.0987880951274, -35.3079468872182, 
                -41.3604221850982, -21.3884575308486,  24.5713233766107, -22.9293036060374, 
                 23.1913501220949, -15.1173367345279,   9.48702774306751, -1.83047533808868, 
                 17.6942952358729,  -2.15702440220209, -7.52962741290285,  5.25282192766975, 
                 -1.11643153921311,-47.5157531063491,  37.306680636605,   35.7280188087715, 
                -26.4286365026548,   6.93701074747048,-13.2741031123528,  30.1503485531617, 
                  4.59960258007609,-13.6490513152672,  22.1929149786996,  14.1864886368628, 
                  3.50672317759581,  3.62912755049426,-26.5924030770213,  23.0986282887491, 
                  0.778370457527042,-4.42495365406214, 13.0419849276326,  4.82280458842396]
@test_approx_eq MixedModels.updateμ!(lmb5) 99435.11862554778
@test_approx_eq pwrss(lmb5) 117889.39251214844
@test_approx_eq scale(lmb5) 25.591816455889482
@test_approx_eq scale(lmb5,true) 654.9410695119358
@test_approx_eq std(lmb5)[1] [23.780564172065286,5.7168335583606575]
@test_approx_eq std(lmb5)[2] [scale(lmb5)]
