using MixedModels, ForwardDiff, Test
include("modelcache.jl")

fm1 = only(models(:dyestuff2))
@test ForwardDiff.gradient(fm1) ≈ [0.0]
@test ForwardDiff.hessian(fm1) ≈  [28.768681]

fm2 = last(models(:sleepstudy))
# not sure what to make of the poor tolerance here
@test ForwardDiff.gradient(fm2) ≈ [0.0, 0.0, 0.0] atol=0.005
@test ForwardDiff.hessian(fm2) ≈ [45.4126 35.9366 6.3549
                                  35.9366 465.7398 203.9920
                                   6.3549 203.9920 963.9520] rtol=1e-6
