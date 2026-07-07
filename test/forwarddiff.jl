using MixedModels, ForwardDiff, Test
include("modelcache.jl")

fm1 = only(models(:dyestuff2))
@test ForwardDiff.gradient(fm1) ≈ [0.0]
@test ForwardDiff.hessian(fm1) ≈ [28.768680] atol=1e-5

fm2 = last(models(:sleepstudy))
# not sure what to make of the poor tolerance here
@test ForwardDiff.gradient(fm2) ≈ [0.0, 0.0, 0.0] atol = 0.005
@test ForwardDiff.hessian(fm2) ≈ [45.4123530453015 35.93768652566969 6.355982998132746;
                                  35.937686525661945 465.7402111242108 203.9973706710023;
                                  6.355982998133106 203.99737067100543 963.9594090304945] rtol = 1e-6

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
