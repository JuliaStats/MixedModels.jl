using MixedModels, ForwardDiff, Test
include("modelcache.jl")

fm1 = only(models(:dyestuff2))
@test ForwardDiff.gradient(fm1) ≈ [0.0]
@test ForwardDiff.hessian(fm1) ≈ [28.768681]

fm2 = last(models(:sleepstudy))
# not sure what to make of the poor tolerance here
@test ForwardDiff.gradient(fm2) ≈ [0.0, 0.0, 0.0] atol = 0.005
@test ForwardDiff.hessian(fm2) ≈ [45.41182543191135 35.937236480602806 6.35599436072896;
                                  35.93723648060227 465.7361238110258 203.9936172832127;
                                  6.355994360727541 203.9936172832205 963.9496683753232] rtol = 1e-6

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
