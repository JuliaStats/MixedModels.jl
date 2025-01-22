using PRIMA
using MixedModels: prfit!
using MixedModels: dataset

include("modelcache.jl")

# model = first(models(:sleepstudy))

@testset "formula($model)" for model in models(:sleepstudy)
    prmodel = prfit!(LinearMixedModel(formula(model), dataset(:sleepstudy)); progress=false)

    @test isapprox(loglikelihood(model), loglikelihood(prmodel))
end
