using MixedModels, ForwardDiff, Test
include("modelcache.jl")

fm1 = only(models(:dyestuff2))
@test ForwardDiff.gradient(fm1) ≈ [0.0]
@test ForwardDiff.hessian(fm1) ≈ [28.768680] atol = 1e-5

fm2 = last(models(:sleepstudy))
# not sure what to make of the poor tolerance here
@test ForwardDiff.gradient(fm2) ≈ [0.0, 0.0, 0.0] atol = 0.005
# reference values are for the σ-profiled objective
@test ForwardDiff.hessian(fm2) ≈ [40.887267 31.517826 -14.529421;
    31.517826 461.423129 183.597634;
    -14.529421 183.597634 867.563441] rtol = 0.001

# REML and zerocorr
fm3 = lmm(
    @formula(reaction ~ 1 + days + zerocorr(1 + days | subj)),
    MixedModels.dataset(:sleepstudy);
    REML=true,
)
@test ForwardDiff.gradient(fm3) ≈ [0.0, 0.0] atol = 0.005

# crossed random effects
if !Sys.iswindows() # this doesn't meet even the very loose tolerance on windows
    fm4 = last(models(:kb07))
    g = ForwardDiff.gradient(fm4)
    @test g ≈ zero(g) atol = 0.5
end
