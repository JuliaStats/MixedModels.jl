fm1 = lmm(Yield ~ 1 + (1|Batch),ds)

@test size(fm1.A) == (2,2)
@test size(fm1.trms) == (2,)
@test size(fm1.R) == (2,2)
@test size(fm1.Λ) == (1,)
@test lowerbd(fm1) == zeros(1)
@test fm1[:θ] == ones(1)

fm1[:θ] = [0.713]
@test objective(fm1) ≈ 327.34216280955366

fit!(fm1);
@test objective(fm1) ≈ 327.3270598811428
@test_approx_eq_eps fm1[:θ] [0.752580] 1.e-5
@test_approx_eq_eps deviance(fm1) 327.32705988 1.e-6
@test_approx_eq fixef(fm1) [1527.5]
@test_approx_eq coef(fm1) [1527.5]

@test lowerbd(fm2) == zeros(1)
fit!(fm2)
@test fm2[:θ] == zeros(1)

fm3 = lmm(Reaction ~ 1+Days + (1+Days|Subject),slp)
@test lowerbd(fm3) == [0.,-Inf,0.]
fit!(fm3)
@test MixedModels.objective(fm3) ≈ 1751.9393444663153
@test_approx_eq_eps fm3[:θ] [0.9292213074888169,0.01816838485113137,0.22264487095998978] 1.e-6

fm4 = lmm(Strength ~ 1 + (1|Sample) + (1|Batch),psts)
@test lowerbd(fm4) == zeros(2)
@test fm4[:θ] == ones(2)
fit!(fm4)
@test MixedModels.objective(fm4) ≈ 247.99446586325791
#@test fm4[:θ] ≈ [3.526885897445589,1.3299228050484744]

fm5 = lmm(Diameter ~ 1 + (1|Plate) + (1|Sample),pen)
@test lowerbd(fm5) == zeros(2)
@test fm5[:θ] == ones(2)
fit!(fm5)
