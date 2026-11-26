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
    @test isapprox(loglikelihood(model), loglikelihood(prmodel)) atol = 1.e-5
end

@testset "θ profile with an off-diagonal θ" begin
    # with θ2 held fixed, the conditional optimizer could make the diagonal element θ1
    # negative and so move to the mirror image of the model at -θ2
    contrasts = Dict{Symbol,Any}(:spkr => EffectsCoding(),
        :prec => EffectsCoding(; base="maintain"), :load => EffectsCoding())
    m = LinearMixedModel(
        @formula(rt_trunc ~ 1 + prec + spkr + load + (1 + prec | item) + (1 | subj)),
        dataset(:kb07); contrasts)
    m.optsum.backend = :prima
    m.optsum.optimizer = :bobyqa
    fit!(m; progress=false)
    pr = @suppress profile(m)
    θ2tbl = filter(r -> r.p == :θ2, pr.tbl)
    @test issorted(getproperty.(θ2tbl, :θ2))
    @test issorted(getproperty.(θ2tbl, :ζ))
    # diagonal elements of λ are non-negative throughout the θ profiles
    @test all(r -> r.θ1 ≥ 0 && r.θ3 ≥ 0 && r.θ4 ≥ 0,
        filter(r -> startswith(string(r.p), 'θ'), pr.tbl))
end

@testset "refit!" begin
    refit!(prmodel; progress=false)
    @test prmodel.optsum.fitlog.θ[begin] == [1.0]
end

@testset "optimization starts from initial" begin
    m = LinearMixedModel(@formula(reaction ~ 1 + days + (1 + days | subj)),
        dataset(:sleepstudy))
    m.optsum.backend = :prima
    m.optsum.optimizer = :bobyqa
    θ₀ = [0.9, 0.02, 0.2]
    copyto!(m.optsum.initial, θ₀)   # m.optsum.final is still the default
    fit!(m; progress=false)
    @test first(m.optsum.fitlog.θ) == θ₀

    gm = GeneralizedLinearMixedModel(@formula(use ~ 1 + urban + (1 | urban & dist)),
        dataset(:contra), Bernoulli())
    copyto!(gm.optsum.initial, [0.5])
    fit!(gm; fast=true, optimizer=:bobyqa, backend=:prima, progress=false)
    @test first(gm.optsum.fitlog.θ) == [0.5]
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
    @test isapprox(loglikelihood(model), loglikelihood(prmodel)) atol = 0.005
    refit!(prmodel; fast=true, progress=false)
    refit!(model; fast=true, progress=false)
    @test isapprox(loglikelihood(model), loglikelihood(prmodel)) atol = 0.005

    optsum = deepcopy(prmodel.optsum)
    optsum.final = [0.2612]
    optsum.finitial = 2595.85
    optsum.fmin = 2486.42
    optsum.feval = 17
    optsum.pirls_ftol_rel = 1e-8

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
    pirls_maxiter:            10
    pirls_ftol_rel:           1.0e-8
    pirls_ftol_abs:           1.0e-5
    pirls_maxhalfstep:        10
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
    | pirls_maxiter            | 10                |
    | pirls_ftol_rel           | 1.0e-8            |
    | pirls_ftol_abs           | 1.0e-5            |
    | pirls_maxhalfstep        | 10                |
    | **Result**               |                   |
    | Function evaluations     | 17                |
    | Final parameter vector   | [0.2612]          |
    | Final objective value    | 2486.42           |
    | Return code              | `SMALL_TR_RADIUS` |
    """

    @test startswith(out, expected)
end
