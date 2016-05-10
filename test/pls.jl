fm1 = lmm(Yield ~ 1 + (1|Batch), ds)
fm1w = lmm(Yield ~ 1 + (1|Batch), ds; wt = ones(size(ds, 1)))

@test size(fm1.A) == (3, 3)
@test size(fm1.wttrms) == (3,)
@test size(fm1.R) == (3, 3)
@test size(fm1.Λ) == (1,)
@test lowerbd(fm1) == zeros(1)
@test fm1[:θ] == ones(1)

fm1[:θ] = [0.713]
@test objective(fm1) ≈ 327.34216280955366
MixedModels.describeblocks(IOBuffer(), fm1)

fit!(fm1);
@test_approx_eq_eps objective(fm1) 327.3270598811428 1e-3
@test_approx_eq_eps fm1[:θ] [0.752580] 1.e-5
@test_approx_eq_eps deviance(fm1) 327.32705988 1.e-3
@test_approx_eq_eps AIC(fm1) 333.3270598811394 1.e-3
@test_approx_eq_eps BIC(fm1) 337.5306520261259 1.e-3
@test fixef(fm1) ≈ [1527.5]
@test MixedModels.fixef!(zeros(1),fm1) ≈ [1527.5]
@test coef(fm1) ≈ [1527.5]
@test cond(fm1) == ones(1)
cm = coeftable(fm1)
@test size(cm.mat) == (1,3)
@test MixedModels.fnames(fm1) == [:Batch]
@test model_response(fm1) == convert(Vector,ds[:Yield])
@test abs(sum(ranef(fm1,true)[1])) < 1.e-5

@test_approx_eq_eps logdet(fm1) 8.06014522999825 1.e-3
@test_approx_eq_eps varest(fm1) 2451.2501089607676 1.e-3
@test_approx_eq_eps pwrss(fm1) 73537.50326882303 1.e-1
@test_approx_eq_eps stderr(fm1) [17.69455188898009] 1.e-4

vc = VarCorr(fm1)
show(IOBuffer(), vc)
@test vc.s == sdest(fm1)
srand(1234321)
simulate!(fm1)   # changes the fit
@test_approx_eq_eps deviance(fm1) 339.0218639362958 1.e-3
bootstrap(fm1,1,(i,x) -> fixef(x)) # bootstrap restores the fit
@test_approx_eq_eps deviance(fm1) 339.0218639362958 1.e-3

fm2 = lmm(Yield ~ 1 + (1|Batch),ds2)
@test lowerbd(fm2) == zeros(1)
#redirect_stdout(IOBuffer())
fit!(fm2,true)
#redirect_stdout()
show(IOBuffer(), fm2)
@test abs(fm2[:θ][1]) < 1.0e-9
@test objective(fm2) ≈ 162.87303665382575
@test abs(std(fm2)[1][1]) < 1.0e-9
@test std(fm2)[2] ≈ [3.653231351374652]
@test stderr(fm2) ≈ [0.6669857396443261]
@test coef(fm2) ≈ [5.6656]
@test logdet(fm2) ≈ 0.0
refit!(fm2,convert(Vector,ds[:Yield]))

fm3 = lmm(Reaction ~ 1+Days + (1+Days|Subject),slp)
@test lowerbd(fm3) == [0.,-Inf,0.]
@test isa(fm3.A[1,1],MixedModels.HBlkDiag{Float64})
@test size(fm3.A[1,1]) == (36,36)
@test fm3.A[1,1][1,1] == 10.
@test fm3.A[1,1][6,1] == 0.
@test_throws BoundsError size(fm3.A[1,1],0)
@test size(fm3.A[1,1],1) == 36
@test full(fm3.A[1,1])[1:2,1:2] == reshape([10.,45,45,285],(2,2))

fit!(fm3)

@test objective(fm3) ≈ 1751.9393444647046
@test_approx_eq_eps fm3[:θ] [0.9292213074888169,0.01816838485113137,0.22264487095998978] 1.e-6
@test pwrss(fm3) ≈ 117889.46144025437
@test_approx_eq_eps logdet(fm3) 73.90322021999222 1e-3
@test_approx_eq_eps stderr(fm3) [6.632257721914501,1.5022354739749826]  1.e-4
@test coef(fm3) ≈ [251.40510484848477,10.4672859595959]
@test fixef(fm3) ≈ [251.40510484848477,10.4672859595959]
@test_approx_eq_eps stderr(fm3) [6.632246393963571,1.502190605041084] 1.e-2
@test_approx_eq_eps std(fm3)[1] [23.780468100188497,5.716827903196682] 1.e-2
@test_approx_eq_eps logdet(fm3) 73.90337187545992 1.e-3
@test_approx_eq diag(cor(fm3)[1]) ones(2)
@test_approx_eq_eps cond(fm3) [4.1752507630514915] 1.e-4
@test loglikelihood(fm3) ≈ -875.9696722323523
@test eltype(fm3.wttrms[1]) === Float64
const u3 = ranef(fm3,true)
@test length(u3) == 1
@test size(u3[1]) == (2,18)
@test_approx_eq_eps u3[1][1,1] 3.030300122575336 1.e-3
const b3 = ranef(fm3)
@test length(b3) == 1
@test size(b3[1]) == (2,18)
@test_approx_eq_eps b3[1][1,1] 2.815819441982976 1.e-3

fm4 = lmm(Reaction ~ Days + (1|Subject) + (0+Days|Subject), slp);
@test size(fm4) == (180,2,36,2)
@test fm4[:θ] == ones(2)
@test lowerbd(fm4) == zeros(2)

fit!(fm4);

@test_approx_eq_eps deviance(fm4) 1752.0032551398835 1.e-3
@test_approx_eq_eps objective(fm4) 1752.0032551398835 1.e-3
@test_approx_eq coef(fm4) [251.40510484848585,10.467285959595715]
@test_approx_eq fixef(fm4) [251.40510484848585,10.467285959595715]
@test_approx_eq_eps stderr(fm4) [6.707710260366577,1.5193083237479683] 1.e-3
@test_approx_eq_eps fm4[:θ] [0.9458106880922268,0.22692826607677266] 1.e-4
@test std(fm4)[1] ≈ [24.171449463289047]
@test std(fm4)[2] ≈ [5.799379721123582]
@test std(fm4)[3] ≈ [25.556130034081047]
@test_approx_eq_eps logdet(fm4) 74.46952585564611 1.e-3

tbl = MixedModels.lrt(fm4,fm3)

@test_approx_eq_eps tbl[:Deviance] [1752.0032551398835,1751.9393444636157] 1e-3
@test tbl[:Df] == [5,6]

simulate!(fm3)  # to test one of the unscaledre methods

fm5 = lmm(Diameter ~ (1|Plate) + (1|Sample), pen);
@test size(fm5) == (144,1,30,2)
@test fm5[:θ] == ones(2)
@test lowerbd(fm5) == zeros(2)

fit!(fm5)

@test_approx_eq_eps objective(fm5) 332.18834867227616 1.e-3
@test_approx_eq_eps coef(fm5) [22.97222222222222] 1.e-3
@test_approx_eq_eps fixef(fm5) [22.97222222222222] 1.e-3
@test coef(fm5)[1] ≈ mean(pen[:Diameter])
@test_approx_eq_eps stderr(fm5) [0.7445960346851368] 1.e-4
@test_approx_eq_eps fm5[:θ] [1.5375772376554968,3.219751321180035] 1.e-3
@test_approx_eq_eps std(fm5)[1] [0.8455645948223015] 1.e-4
@test_approx_eq_eps std(fm5)[2] [1.770647779277388] 1.e-4
@test_approx_eq_eps varest(fm5) 0.3024263987592062 1.e-4
@test_approx_eq_eps logdet(fm5) 95.74614821367786 1.e-3
@test length(ranef(fm5)) == 2

fm6 = lmm(Strength ~ (1|Sample) + (1|Batch), psts);
@test size(fm6) == (60,1,40,2)
@test fm6[:θ] == ones(2)
@test lowerbd(fm6) == zeros(2)

fit!(fm6);

@test_approx_eq_eps objective(fm6) 247.99446586289676 1.e-3
@test_approx_eq_eps coef(fm6) [60.05333333333329] 1.e-3
@test_approx_eq_eps fixef(fm6) [60.05333333333329] 1.e-3
@test_approx_eq_eps stderr(fm6) [0.6421359883527029] 1.e-4
@test_approx_eq_eps fm6[:θ] [3.5268858714382905,1.3299230213750168] 1.e-3
@test_approx_eq_eps std(fm6)[1] [2.904069002535747] 1.e-3
@test_approx_eq_eps std(fm6)[2] [1.095070371687089] 1.e-4
@test_approx_eq_eps std(fm6)[3] [0.8234088395243269] 1.e-4
@test_approx_eq_eps varest(fm6) 0.6780020742644107 1.e-4
@test_approx_eq_eps logdet(fm6) 101.0381339953986 1.e-3
