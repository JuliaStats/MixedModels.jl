using MixedModels, FiniteDiff, Test
include("modelcache.jl")

fm1 = only(models(:dyestuff2))
@test FiniteDiff.finite_difference_gradient(fm1) ≈ [0.0]
@test FiniteDiff.finite_difference_hessian(fm1) ≈  [28.7686] atol=0.0001

fm2 = last(models(:sleepstudy))
@test FiniteDiff.finite_difference_gradient(fm2) ≈ [0.0, 0.0, 0.0] atol=0.005

# REML and zerocorr
fm3 = lmm(@formula(reaction ~ 1 + days + zerocorr(1+days|subj)), MixedModels.dataset(:sleepstudy); REML=true)
@test FiniteDiff.finite_difference_gradient(fm3) ≈ [0.0,0.0] atol=0.001

# crossed random effects
fm4 = last(models(:kb07))
g = FiniteDiff.finite_difference_gradient(fm4)
@test g ≈ zero(g) atol=0.1
