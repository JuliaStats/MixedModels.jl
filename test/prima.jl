using PRIMA
using MixedModels: prfit!
using MixedModels: dataset

include("modelcache.jl")

# model = first(models(:sleepstudy))

@testset "formula($model)" for model in models(:sleepstudy)
    prmodel = prfit!(LinearMixedModel(formula(model), dataset(:sleepstudy)); progress=false)

    @test isapprox(loglikelihood(model), loglikelihood(prmodel))
    @test prmodel.optsum.optimizer == :bobyqa
    @test prmodel.optsum.backend == :prima
end

model = first(models(:sleepstudy))
prmodel = LinearMixedModel(formula(model), dataset(:sleepstudy))
prmodel.optsum.backend = :prima

@testset "$optimizer" for optimizer in (:cobyla, :lincoa)
    MixedModels.unfit!(prmodel)
    prmodel.optsum.optimizer = optimizer
    fit!(prmodel)
    @test isapprox(loglikelihood(model), loglikelihood(prmodel))
end
