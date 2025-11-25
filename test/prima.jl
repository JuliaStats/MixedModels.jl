using PRIMA
using MixedModels: unfit!, dataset
using Suppressor

include("modelcache.jl")

@testset "$(formula(model))" for model in models(:sleepstudy)
    prmodel = LinearMixedModel(formula(model), dataset(:sleepstudy))
    prmodel.optsum.backend = :prima
    prmodel.optsum.optimizer = :bobyqa
    fit!(prmodel; progress=false)

    @test isapprox(loglikelihood(model), loglikelihood(prmodel))
    @test prmodel.optsum.optimizer == :bobyqa
    @test prmodel.optsum.backend == :prima

    @testset "profile" begin
        profile_prima = @suppress profile(prmodel)
        profile_nlopt = @suppress profile(model)
        @test isapprox(profile_prima.tbl.ζ, profile_nlopt.tbl.ζ; rtol=0.0001)
    end
end

model = first(models(:sleepstudy))
prmodel = LinearMixedModel(formula(model), dataset(:sleepstudy))
prmodel.optsum.backend = :prima

@testset "$optimizer" for optimizer in (:cobyla, :lincoa, :newuoa)
    unfit!(prmodel)
    prmodel.optsum.optimizer = optimizer
    fit!(prmodel; progress=false)
    @test isapprox(loglikelihood(model), loglikelihood(prmodel)) atol=1.e-5
end

@testset "refit!" begin
    refit!(prmodel; progress=false)
    @test prmodel.optsum.fitlog.θ[begin] == [1.0]
end

@testset "failure" begin
    unfit!(prmodel)
    prmodel.optsum.optimizer = :bobyqa
    prmodel.optsum.maxfeval = 5
    @test_logs((:warn, r"PRIMA optimization failure"),
        fit!(prmodel; progress=false))
end

@testset "GLMM + optsum show" begin
    model = fit(MixedModel,
        @formula(use ~ 1 + age + abs2(age) + urban + livch + (1 | urban & dist)),
        dataset(:contra), Binomial(); progress=false)
    prmodel = unfit!(deepcopy(model))
    fit!(prmodel; optimizer=:bobyqa, backend=:prima, progress=false)
    @test isapprox(loglikelihood(model), loglikelihood(prmodel)) atol=0.001
    refit!(prmodel; fast=true, progress=false)
    refit!(model; fast=true, progress=false)
    @test isapprox(loglikelihood(model), loglikelihood(prmodel)) atol=0.001

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
