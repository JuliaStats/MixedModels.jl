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

@testset "optsum show" begin
    model = first(models(:contra))
    prmodel = unfit!(deepcopy(model))
    fit!(prmodel; optimizer=:bobyqa, backend=:prima)
    @test isapprox(loglikelihood(model), loglikelihood(prmodel))

    optsum = deepcopy(prmodel.optsum)
    optsum.final = [0.2612]
    optsum.finitial = 2595.85
    optsum.fmin = 2486.42
    optsum.feval = 17

    out = sprint(show, MIME("text/plain"), optsum)
    expected = """
    Initial parameter vector: [1.0]
    Initial objective value:  2595.85

    Backend:                  prima
    Optimizer:                bobyqa
    Lower bounds:             [0.0]
    rhobeg:                   1.0
    rhoend:                   1.0e-6
    maxfeval:                 -1

    Function evaluations:     17
    xtol_zero_abs:            0.001
    ftol_zero_abs:            1.0e-5
    Final parameter vector:   [0.2612]
    Final objective value:    2486.42
    Return code:              SMALL_TR_RADIUS
    """

    @test startswith(out, expected)

    out = sprint(show, MIME("text/markdown"), optsum)
    expected = """
    |                          |                   |
    |:------------------------ |:----------------- |
    | **Initialization**       |                   |
    | Initial parameter vector | [1.0]             |
    | Initial objective value  | 2595.85           |
    | **Optimizer settings**   |                   |
    | Optimizer                | `bobyqa`          |
    | Backend                  | `prima`           |
    | Lower bounds             | [0.0]             |
    | rhobeg                   | 1.0               |
    | rhoend                   | 1.0e-6            |
    | maxfeval                 | -1                |
    | xtol_zero_abs            | 0.001             |
    | ftol_zero_abs            | 1.0e-5            |
    | **Result**               |                   |
    | Function evaluations     | 17                |
    | Final parameter vector   | [0.2612]          |
    | Final objective value    | 2486.42           |
    | Return code              | `SMALL_TR_RADIUS` |
    """

    @test startswith(out, expected)
end
