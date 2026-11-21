using DataFrames
using Distributions
using MixedModels
using PooledArrays
using StableRNGs
using Tables
using Test

using GLM: Link
using MixedModelsDatasets: dataset

include("modelcache.jl")

@testset "GLMM from MixedModel" begin
    f = first(gfms[:contra])
    d = dataset(:contra)
    @test MixedModel(f, d, Bernoulli()) isa GeneralizedLinearMixedModel
    @test MixedModel(f, d, Bernoulli(), ProbitLink()) isa GeneralizedLinearMixedModel
end

@testset "Type for instance" begin
    vaform = @formula(r2 ~ 1 + anger + gender + btype + situ + (1 | subj) + (1 | item))
    verbagg = dataset(:verbagg)
    @test_throws ArgumentError fit(MixedModel, vaform, verbagg, Bernoulli, LogitLink)
    @test_throws ArgumentError fit(MixedModel, vaform, verbagg, Bernoulli(), LogitLink)
    @test_throws ArgumentError fit(MixedModel, vaform, verbagg, Bernoulli, LogitLink())
    @test_throws ArgumentError fit(
        GeneralizedLinearMixedModel, vaform, verbagg, Bernoulli, LogitLink
    )
    @test_throws ArgumentError fit(
        GeneralizedLinearMixedModel, vaform, verbagg, Bernoulli(), LogitLink
    )
    @test_throws ArgumentError fit(
        GeneralizedLinearMixedModel, vaform, verbagg, Bernoulli, LogitLink()
    )
    @test_throws ArgumentError GeneralizedLinearMixedModel(
        vaform, verbagg, Bernoulli, LogitLink
    )
    @test_throws ArgumentError GeneralizedLinearMixedModel(
        vaform, verbagg, Bernoulli(), LogitLink
    )
    @test_throws ArgumentError GeneralizedLinearMixedModel(
        vaform, verbagg, Bernoulli, LogitLink()
    )
end

@testset "contra" begin
    contra = dataset(:contra)
    gm0 = fit(
        MixedModel,
        first(gfms[:contra]),
        contra,
        Bernoulli();
        fast=true,
        progress=false,
    )
    fitlog = gm0.optsum.fitlog
    @test length(fitlog) == gm0.optsum.feval
    @test fitlog.θ[begin] == gm0.optsum.initial
    @test fitlog.objective[begin] ≈ gm0.optsum.finitial
    # XXX this should be exact equality and it is indeed when stepping through manually
    # but not when run via Pkg.test(). I have no idea why.
    @test fitlog.θ[end] ≈ gm0.optsum.final
    @test fitlog.objective[end] ≈ gm0.optsum.fmin
    @test isapprox(gm0.θ, [0.5720746212924732], atol=0.001)
    @test !issingular(gm0)
    @test issingular(gm0, [0])
    @test isapprox(deviance(gm0), 2361.657202855648, atol=0.001)
    # the first 9 BLUPs -- I don't think there's much point in testing all 102
    blups = [-0.5853637711570235, -0.9546542393824562, -0.034754249031292345, # values are the same but in different order
        0.2894692928724314, 0.6381376605845264, -0.2513134928312374,
        0.031321447845204374, 0.10836110432794945, 0.24632286640099466]
    @test only(ranef(gm0))[1:9] ≈ blups atol = 1e-4
    retbl = raneftables(gm0)
    @test isone(length(retbl))
    @test isa(retbl, NamedTuple)
    @test Tables.istable(only(retbl))
    @test !dispersion_parameter(gm0)
    @test dispersion(gm0, false) == 1
    @test dispersion(gm0, true) == 1
    @test sdest(gm0) === missing
    @test varest(gm0) === missing
    @test gm0.σ === missing
    @test Distribution(gm0) == Distribution(gm0.resp)
    @test Link(gm0) == Link(gm0.resp)

    gm1 = fit(MixedModel, first(gfms[:contra]), contra, Bernoulli(); progress=false)
    @test isapprox(gm1.θ, [0.5730523416716424], atol=0.005)
    @test isapprox(deviance(gm1), 2361.545768866505, rtol=0.00001)
    @test isapprox(loglikelihood(gm1), -1180.772884433253, rtol=0.00001)

    @test dof(gm0) == length(gm0.β) + length(gm0.θ)
    @test nobs(gm0) == 1934
    refit!(gm0; fast=false, nAGQ=7, progress=false)  # changed to fast=false; fast=true and nAGQ > 0 contradict
    @test deviance(gm0) ≈ 2360.8760880739255 atol = 0.005
    gm1 = fit(MixedModel, first(gfms[:contra]), contra, Bernoulli(); nAGQ=7, progress=false)
    @test deviance(gm1) ≈ 2360.8760880739255 atol = 0.005
    @test deviance(gm0) ≈ deviance(gm1) atol = 0.005
    @test gm1.β == gm1.beta
    @test gm1.θ == gm1.theta
    gm1y = gm1.y
    @test length(gm1y) == size(gm1.X, 1)
    @test eltype(gm1y) == eltype(gm1.X)
    @test gm1y == (MixedModels.dataset(:contra).use .== "Y")
    @test response(gm1) == gm1y
    @test !islinear(gm1)
    @test :θ in propertynames(gm0)

    @testset "GLMM rePCA" begin
        @test length(MixedModels.PCA(gm0)) == 1
        @test length(MixedModels.rePCA(gm0)) == 1
        @test length(gm0.rePCA) == 1
    end
    # the next three values are not well defined in the optimization
    #@test isapprox(logdet(gm1), 75.7217, atol=0.1)
    #@test isapprox(sum(abs2, gm1.u[1]), 48.4747, atol=0.1)
    #@test isapprox(sum(gm1.resp.devresid), 2237.349, atol=0.1)
    show(IOBuffer(), gm1)
    show(IOBuffer(), BlockDescription(gm0))

    gm_slope = fit(MixedModel, gfms[:contra][2], contra, Bernoulli(); progress=false)
    @test !issingular(gm_slope)
    @test issingular(gm_slope, zeros(5))
end

@testset "cbpp" begin
    cbpp = dataset(:cbpp)
    gm2 = fit(
        MixedModel,
        first(gfms[:cbpp]),
        cbpp,
        Binomial();
        weights=float(cbpp.hsz),
        progress=false,
        init_from_lmm=[:β, :θ],
    )
    @test weights(gm2) == cbpp.hsz
    @test deviance(gm2, true) ≈ 100.09585620707632 rtol = 0.0001
    @test sum(abs2, gm2.u[1]) ≈ 9.72301224524056 rtol = 0.0001
    @test logdet(gm2) ≈ 16.901127982275217 rtol = 0.0001
    @test isapprox(sum(gm2.resp.devresid), 73.47171597956056, atol=0.001)
    @test isapprox(loglikelihood(gm2), -92.02628187247377, atol=0.001)
    @test !dispersion_parameter(gm2)
    @test dispersion(gm2, false) == 1
    @test dispersion(gm2, true) == 1
    @test sdest(gm2) === missing
    @test varest(gm2) === missing
    @test gm2.σ === missing

    @testset "GLMM refit" begin
        gm2r = deepcopy(gm2)
        @test_throws ArgumentError fit!(gm2r; progress=false)

        refit!(gm2r; fast=true, progress=false)
        @test length(gm2r.optsum.final) == 1
        @test gm2r.θ ≈ gm2.θ atol = 1e-3

        # swapping successes and failures to give us the same model
        # but with opposite signs. healthy ≈ 1 - response(gm2r)
        # but defining it in terms of the original values avoids some
        # nasty floating point issues
        healthy = @. (cbpp.hsz - cbpp.incid) / cbpp.hsz
        refit!(gm2r, healthy; fast=false, progress=false)
        @test length(gm2r.optsum.final) == 5
        @test gm2r.β ≈ -gm2.β atol = 1e-3
        # @test gm2r.θ ≈ gm2.θ atol=1e-3    # in gm2r θ[1] is negative.  Can't work out why.
    end

    @testset "constant response" begin
        cbconst = DataFrame(cbpp)
        cbconst.incid = zero(cbconst.incid)
        # we do construction and fitting in two separate steps to make sure
        # that construction succeeds and that the ArgumentError occurs in fitting.
        mcbconst = GeneralizedLinearMixedModel(
            first(gfms[:cbpp]), cbconst, Binomial(); weights=float(cbpp.hsz)
        )
        @test mcbconst isa GeneralizedLinearMixedModel
        @test_throws ArgumentError(
            "The response is constant and thus model fitting has failed"
        ) fit!(mcbconst; progress=false)
    end
end

@testset "verbagg" begin
    gm3 = fit(
        MixedModel, only(gfms[:verbagg]), dataset(:verbagg), Bernoulli(); progress=false
    )
    @test deviance(gm3) ≈ 8151.40 rtol = 1e-5
    @test fitted(gm3) == predict(gm3)
    # these two values are not well defined at the optimum
    @test isapprox(sum(x -> sum(abs2, x), gm3.u), 273.29646346940785, rtol=1e-3)
    @test sum(gm3.resp.devresid) ≈ 7156.550941446312 rtol = 1e-4
end

@testset "grouseticks" begin
    center(v::AbstractVector) = v .- (sum(v) / length(v))
    grouseticks = DataFrame(dataset(:grouseticks))
    grouseticks.ch = center(grouseticks.height)
    gm4 = fit(
        MixedModel,
        only(gfms[:grouseticks]),
        grouseticks,
        Poisson();
        fast=true,
        progress=false,
    )
    @test isapprox(deviance(gm4), 851.4046, atol=0.001)
    # these two values are not well defined at the optimum
    #@test isapprox(sum(x -> sum(abs2, x), gm4.u), 196.8695297987013, atol=0.1)
    #@test isapprox(sum(gm4.resp.devresid), 220.92685781326136, atol=0.1)
    @test !dispersion_parameter(gm4)
    @test dispersion(gm4, false) == 1
    @test dispersion(gm4, true) == 1
    @test sdest(gm4) === missing
    @test varest(gm4) === missing
    @test gm4.σ === missing
    gm4slow = fit(
        MixedModel,
        only(gfms[:grouseticks]),
        grouseticks,
        Poisson();
        fast=false,
        progress=false,
    )
    # this tolerance isn't great, but then again the optimum isn't well defined
    # @test gm4.θ ≈ gm4slow.θ rtol=0.05
    # @test gm4.β[2:end] ≈ gm4slow.β[2:end] atol=0.1
    @test isapprox(deviance(gm4), deviance(gm4slow); rtol=0.1)
end

# @testset "goldstein" begin # from a 2020-04-22 msg by Ben Goldstein to R-SIG-Mixed-Models
#     goldstein = (
#         group=PooledArray(repeat(string.('A':'J'); outer=10)),
#         y=[
#             83, 3, 8, 78, 901, 21, 4, 1, 1, 39,
#             82, 3, 2, 82, 874, 18, 5, 1, 3, 50,
#             87, 7, 3, 67, 914, 18, 0, 1, 1, 38,
#             86, 13, 5, 65, 913, 13, 2, 0, 0, 48,
#             90, 5, 5, 71, 886, 19, 3, 0, 2, 32,
#             96, 1, 1, 87, 860, 21, 3, 0, 1, 54,
#             83, 2, 4, 70, 874, 19, 5, 0, 4, 36,
#             100, 11, 3, 71, 950, 21, 6, 0, 1, 40,
#             89, 5, 5, 73, 859, 29, 3, 0, 2, 38,
#             78, 13, 6, 100, 852, 24, 5, 0, 1, 39,
#         ],
#     )
#     gform = @formula(y ~ 1 + (1 | group))
#     m1 = GeneralizedLinearMixedModel(gform, goldstein, Poisson())
#     @test !isfitted(m1)
#     fit!(m1; progress=false)
#     @test isfitted(m1)
#     @test deviance(m1) ≈ 191.25588670286234 rtol = 1.e-5
#     @test only(m1.β) ≈ 4.191646454847604 atol = 1.e-5
#     @test only(m1.θ) ≈ 2.1169067020826726 atol = 1.e-5
#     m11 = fit(MixedModel, gform, goldstein, Poisson(); nAGQ=11, progress=false)
#     @test deviance(m11) ≈ 191.20306323744958 rtol = 1.e-5
#     @test only(m11.β) ≈ 4.191646454847604 atol = 1.e-5
#     @test only(m11.θ) ≈ 2.1169067020826726 atol = 1.e-5
# end

@testset "dispersion" begin
    form = @formula(reaction ~ 1 + days + (1 + days | subj))
    dat = dataset(:sleepstudy)

    # Neither constructing nor fitting a dispersion-family model emits anything:
    # how ϕ was estimated is reported by `show` instead, so it does not repeat
    # once per replicate when `parametricbootstrap` refits in a loop.
    @test_logs GeneralizedLinearMixedModel(form, dat, Gamma())
    @test_logs fit(MixedModel, form, dat, Gamma(), LogLink(); progress=false)

    @testset "show reports how ϕ was estimated" begin
        gmfree = fit(MixedModel, form, dat, Gamma(), LogLink(); progress=false)
        gmplug = fit(MixedModel, form, dat, Gamma(), LogLink(); fast=true, progress=false)
        @test occursin("estimated jointly with β and θ", sprint(show, gmfree))
        @test occursin("Pearson moment estimator", sprint(show, gmplug))
        # families without a dispersion parameter say nothing about ϕ
        gmb = fit(MixedModel, first(gfms[:contra]), dataset(:contra), Bernoulli();
            fast=true, progress=false)
        @test !occursin("Dispersion parameter", sprint(show, gmb))
    end

    # Minimise the Laplace deviance over ϕ alone, holding β and θ (and hence the
    # conditional modes, which do not depend on ϕ) at their fitted values.  This
    # is the conditional MLE of ϕ, found by golden section on log ϕ so that no
    # family-specific score equation is needed.
    function condmle_ϕ(gm)
        uss = sum(u -> sum(abs2, u), gm.u)
        ld = logdet(gm)
        obj(ϕ) = -2 * MixedModels._loglik_data(gm.resp, ϕ) + uss / ϕ + ld
        r = (sqrt(5) - 1) / 2
        a = log(MixedModels.pwrss(gm) / nobs(gm)) - 4
        b = a + 8
        for _ in 1:300
            c, d = b - r * (b - a), a + r * (b - a)
            obj(exp(c)) < obj(exp(d)) ? (b = d) : (a = c)
        end
        return exp((a + b) / 2)
    end

    @testset "normalisation of the u penalty" begin
        # `_laplace_deviance` carries the random-effects penalty as `uss/ϕ`,
        # because θ is relative to the scale parameter (Var(b) = ϕΛΛ', so
        # u ~ N(0, ϕI)) exactly as in LinearMixedModel.  The check that pins
        # this down: for a Normal family, pwrss/n is the conditional MLE of ϕ,
        # so profiling the objective over ϕ at fixed β and θ has to return
        # pwrss/n.  With the penalty left as `uss` it misses by ~16%.
        gm = fit(MixedModel, form, dat, Normal(), SqrtLink(); progress=false)
        # the tolerance is set by how tightly PIRLS converged (`pwrss` comes
        # from the linearised problem, and only coincides with Σ(y-μ)² + uss at
        # the exact PIRLS fixed point), not by the identity itself -- which is
        # exact for any β and θ.  16% vs 1e-4 is still three orders of margin.
        @test condmle_ϕ(gm) ≈ MixedModels.pwrss(gm) / nobs(gm) rtol = 1.0e-4
    end

    @testset "Gamma + LogLink" begin
        gm = fit(MixedModel, form, dat, Gamma(), LogLink(); progress=false)
        @test dispersion_parameter(gm)
        # ϕ is a free parameter of the outer optimisation for fast=false, so the
        # parameter vector is [β; θ; log ϕ]
        @test length(gm.optsum.final) == length(gm.β) + length(gm.θ) + 1
        @test exp(last(gm.optsum.final)) ≈ dispersion(gm, true)
        # Self-consistency: deviance == -2 * loglikelihood, both at the same ϕ.
        @test deviance(gm) ≈ -2 * loglikelihood(gm) atol = 1.0e-8
        # σ accessors agree
        @test sdest(gm) === gm.σ
        @test sdest(gm) ≈ sqrt(varest(gm))
        @test varest(gm) ≈ dispersion(gm, true)
        @test sdest(gm) ≈ dispersion(gm, false)
        # Regression refs for the joint (β, θ, ϕ) optimum.
        @test deviance(gm) ≈ 1732.0017 rtol = 1.0e-4
        @test loglikelihood(gm) ≈ -866.0008 rtol = 1.0e-4
        @test gm.β ≈ [5.532039, 0.033836] rtol = 1.0e-3
        @test gm.θ ≈ [1.247843, -0.006328, 0.216549] rtol = 1.0e-2
        @test sdest(gm) ≈ 0.080848 rtol = 1.0e-3
        # ϕ̂ itself only has a loose regression lock: the joint (β, θ, ϕ) optimum
        # wanders by ~0.15% between runs -- and even between a fit and its own
        # refit! -- because β and θ do, and ϕ̂ follows them.
        @test dispersion(gm, true) ≈ 0.0065364 rtol = 1.0e-2

        # What is tight, and what actually distinguishes this estimator from the
        # fast=true one, is *conditional* on the fitted β and θ: the free
        # parameter is the conditional MLE, which for Gamma solves a digamma
        # equation rather than the moment condition pwrss/n.  Comparing within
        # a single fit sidesteps the scatter above.
        mle = condmle_ϕ(gm)
        moment = MixedModels.pwrss(gm) / nobs(gm)
        @test dispersion(gm, true) ≈ mle rtol = 1.0e-4
        # ...and the moment estimator is a genuinely different answer, about
        # 0.18% away, worth ~3e-4 in deviance -- some five orders of magnitude
        # above what `ftol_rel` can resolve, so this gap is signal, not noise.
        @test !isapprox(moment, mle; rtol=1.0e-3)
        @test moment ≈ mle rtol = 1.0e-2
    end

    @testset "Gamma + LogLink, fast=true plugs ϕ in" begin
        gm = fit(MixedModel, form, dat, Gamma(), LogLink(); fast=true, progress=false)
        # θ only: ϕ is not a free parameter here
        @test length(gm.optsum.final) == length(gm.θ)
        @test isnothing(gm.ϕ[])
        # and `dispersion` falls back to the moment estimator
        @test dispersion(gm, true) ≈ MixedModels.pwrss(gm) / nobs(gm)
        @test deviance(gm) ≈ -2 * loglikelihood(gm) atol = 1.0e-8
        @test deviance(gm) ≈ 1732.0019 rtol = 1.0e-4
        @test gm.β ≈ [5.532041, 0.033835] rtol = 1.0e-3
        @test gm.θ ≈ [1.247817, -0.006347, 0.216551] rtol = 1.0e-2
        @test sdest(gm) ≈ 0.080777 rtol = 1.0e-3
    end

    @testset "Normal + SqrtLink" begin
        gm = fit(MixedModel, form, dat, Normal(), SqrtLink(); progress=false)
        @test dispersion_parameter(gm)
        @test deviance(gm) ≈ -2 * loglikelihood(gm) atol = 1.0e-8
        @test sdest(gm) ≈ sqrt(varest(gm))
        @test deviance(gm) ≈ 1751.9094 rtol = 1.0e-4
        @test gm.β ≈ [15.880180, 0.297900] rtol = 1.0e-3
        @test sdest(gm) ≈ 25.531009 rtol = 1.0e-3
        # Normal is the case where the moment estimator *is* the conditional
        # MLE, so the free parameter has to reproduce pwrss/n.
        @test dispersion(gm, true) ≈ MixedModels.pwrss(gm) / nobs(gm) rtol = 1.0e-3
    end

    @testset "ϕ fixed a priori" begin
        # mirrors `σ` for LinearMixedModel: it lives in `optsum.sigma` and means
        # "do not estimate this", so ϕ = σ² exactly and drops out of the
        # parameter vector
        free = fit(MixedModel, form, dat, Gamma(), LogLink(); progress=false)
        gm = GeneralizedLinearMixedModel(form, dat, Gamma(), LogLink(); σ=0.09)
        fit!(gm; progress=false)

        @test gm.optsum.sigma == 0.09
        @test dispersion(gm, true) == 0.09^2
        @test sdest(gm) == 0.09
        @test varest(gm) == 0.09^2
        @test isnothing(gm.ϕ[])
        # β and θ only -- no trailing log ϕ
        @test length(gm.optsum.final) == length(gm.β) + length(gm.θ)
        # a constrained fit cannot beat the one that optimises over ϕ too
        @test deviance(gm) > deviance(free)
        @test deviance(gm) ≈ -2 * loglikelihood(gm) atol = 1.0e-8
        @test occursin("fixed a priori", sprint(show, gm))

        # the regime survives a refit
        refit!(gm; progress=false)
        @test dispersion(gm, true) == 0.09^2
        @test isnothing(gm.ϕ[])

        # fixing σ is meaningless without a dispersion parameter
        @test_throws ArgumentError GeneralizedLinearMixedModel(
            first(gfms[:contra]), dataset(:contra), Bernoulli(); σ=1.0
        )
    end

    @testset "ϕ regime survives refit! and saveoptsum" begin
        gm = fit(MixedModel, form, dat, Gamma(), LogLink(); progress=false)
        nfree = length(gm.optsum.final)
        ϕfree = dispersion(gm, true)

        io = IOBuffer()
        saveoptsum(io, gm)
        seekstart(io)
        gm2 = GeneralizedLinearMixedModel(form, dat, Gamma(), LogLink())
        restoreoptsum!(gm2, io)
        @test dispersion(gm2, true) ≈ ϕfree
        @test gm2.optsum.fmin ≈ gm.optsum.fmin

        # toggling `fast` moves between the two regimes cleanly
        refit!(gm; fast=true, progress=false)
        @test isnothing(gm.ϕ[])
        @test length(gm.optsum.final) == length(gm.θ)
        refit!(gm; fast=false, progress=false)
        @test !isnothing(gm.ϕ[])
        @test length(gm.optsum.final) == nfree
        # loose: the joint optimum moves by ~0.15% between a fit and its refit
        @test dispersion(gm, true) ≈ ϕfree rtol = 1.0e-2
    end

    @testset "constructor no longer warns" begin
        # Issue #786 - warning moved to fit-time and weakened
        @test_logs GeneralizedLinearMixedModel(form, dat, InverseGaussian())
        @test_logs GeneralizedLinearMixedModel(form, dat, Normal(), SqrtLink())
    end

    @testset "non-dispersion family unaffected" begin
        # Bit-identical invariant: the optimisation objective for Bernoulli
        # is unchanged from the pre-fix form `sum(devresid) + uss + logdet`.
        gm = fit(MixedModel,
            @formula(use ~ 1 + urban + livch * age + (1 | dist)),
            dataset(:contra), Bernoulli(); progress=false)
        @test !dispersion_parameter(gm)
        @test dispersion(gm) == 1
        @test sdest(gm) === missing
        @test varest(gm) === missing
        @test deviance(gm) ≈ 2403.4078 rtol = 1.0e-6
    end

    @testset "AGQ with nAGQ=1 == Laplace" begin
        # The Laplace approximation is AGQ with n=1 by construction, so
        # `_agq_deviance(m, 1)` and `_laplace_deviance(m)` must agree.
        # Tests one dispersion and one non-dispersion family.
        scalar_re = @formula(reaction ~ 1 + days + (1 | subj))

        gm_d = fit(MixedModel, scalar_re, dat, Gamma(), LogLink(); progress=false)
        @test MixedModels._agq_deviance(gm_d, 1) ≈ MixedModels._laplace_deviance(gm_d) atol =
            1.0e-9

        mmec = dataset(:mmec)
        gm_p = fit(MixedModel,
            @formula(deaths ~ 1 + uvb + (1 | region)),
            mmec, Poisson(); offset=log.(mmec.expected), progress=false)
        @test MixedModels._agq_deviance(gm_p, 1) ≈ MixedModels._laplace_deviance(gm_p) atol =
            1.0e-9
    end

    @testset "Cϕ μ-invariance" begin
        # `_agq_deviance` profiles Cϕ once at the mode, relying on the
        # identity that -2·loglik_obs - devresid/ϕ depends only on (y, w, ϕ),
        # not on μ. Smoke test by shifting β (which propagates to μ via
        # updateη!) and checking that Cϕ is unchanged.
        scalar_re = @formula(reaction ~ 1 + days + (1 | subj))
        gm = fit(MixedModel, scalar_re, dat, Gamma(), LogLink(); progress=false)
        r = gm.resp
        ϕ = MixedModels._dispersion(gm)
        nu = length(first(gm.u))
        Cϕ(rr) = -2 * MixedModels._loglik_data(rr, ϕ) - sum(rr.devresid) / ϕ + nu * log(ϕ)
        Cϕ_at_mode = Cϕ(r)

        β_orig = copy(gm.β)
        gm.β[1] += 0.1
        MixedModels.updateη!(gm)
        Cϕ_perturbed = Cϕ(r)
        @test Cϕ_perturbed ≈ Cϕ_at_mode atol = 1.0e-9

        # Restore the model state so this test doesn't leak side effects.
        copyto!(gm.β, β_orig)
        MixedModels.updateη!(gm)
    end

    @testset "Gamma + LogLink, nAGQ > 1" begin
        # Use scalar RE so AGQ is well-defined. NB: lme4 hits a singular fit
        # (θ → 0) on this configuration; our optimiser finds a non-degenerate
        # interior optimum, so no tight lme4 cross-check here.
        scalar_re = @formula(reaction ~ 1 + days + (1 | subj))
        gm = fit(MixedModel, scalar_re, dat, Gamma(), LogLink();
            nAGQ=5, progress=false)
        @test dispersion_parameter(gm)
        @test gm.optsum.nAGQ == 5

        # Regression refs (captured from this fit). loglikelihood is Laplace-
        # only, so it differs from deviance by the AGQ correction.
        @test deviance(gm) ≈ 1767.1545 rtol = 1.0e-4
        @test gm.β ≈ [5.533943, 0.033845] rtol = 1.0e-3
        @test only(gm.θ) ≈ 1.289160 rtol = 1.0e-2
        @test sdest(gm) ≈ 0.096642 rtol = 1.0e-3

        # Laplace logLik at the AGQ-converged params should agree with
        # `_laplace_deviance` (which uses the same ϕ̂ = pwrss/n).
        @test -2 * loglikelihood(gm) ≈ MixedModels._laplace_deviance(gm) atol = 1.0e-8
    end
end

@testset "mmec" begin
    # Data on "Malignant melanoma in the European community" from the mlmRev package for R
    # The offset of log.(expected) is to examine the ratio of observed to expected, based on population
    mmec = dataset(:mmec)
    mmform = @formula(deaths ~ 1 + uvb + (1 | region))
    gm5 = fit(
        MixedModel,
        mmform,
        mmec,
        Poisson();
        offset=log.(mmec.expected),
        nAGQ=11,
        progress=false,
    )
    @test isapprox(deviance(gm5), 655.2533533016059, atol=5.e-3)
    @test isapprox(first(gm5.θ), 0.4121684550775567, atol=1.e-3)
    @test isapprox(first(gm5.β), -0.13860166843315044, atol=1.e-3)
    @test isapprox(last(gm5.β), -0.034414458364713504, atol=1.e-3)
end

@testset "GLMM saveoptsum" begin
    cbpp = dataset(:cbpp)
    gm_original = GeneralizedLinearMixedModel(
        first(gfms[:cbpp]), cbpp, Binomial(); weights=cbpp.hsz
    )
    gm_restored = GeneralizedLinearMixedModel(
        first(gfms[:cbpp]), cbpp, Binomial(); weights=cbpp.hsz
    )
    fit!(gm_original; progress=false, nAGQ=1)

    io = IOBuffer()

    saveoptsum(seekstart(io), gm_original)
    restoreoptsum!(gm_restored, seekstart(io))
    @test gm_original.optsum == gm_restored.optsum
    @test deviance(gm_original) ≈ deviance(gm_restored)

    refit!(gm_original; progress=false, nAGQ=3)
    saveoptsum(seekstart(io), gm_original)
    restoreoptsum!(gm_restored, seekstart(io))
    @test gm_original.optsum == gm_restored.optsum
    @test deviance(gm_original) ≈ deviance(gm_restored)

    refit!(gm_original; progress=false, fast=true)
    saveoptsum(seekstart(io), gm_original)
    restoreoptsum!(gm_restored, seekstart(io))
    @test gm_original.optsum == gm_restored.optsum
    @test deviance(gm_original) ≈ deviance(gm_restored)
end

@testset "Bad initial value" begin
    rng = StableRNG(0)
    df = allcombinations(DataFrame,
        "subject" => 1:10,
        "session" => 1:6,
        "serialpos" => 1:12)
    df[!, :recalled] = rand(rng, [0, 1], nrow(df))

    form = @formula(
        recalled ~ serialpos + zerocorr(serialpos | subject) + (1 | subject & session)
    )
    glmm = @test_logs((:warn, r"Evaluation at default initial parameter vector failed"),
        GeneralizedLinearMixedModel(form, df, Bernoulli()))
    @test all(==(1e-8), glmm.optsum.initial)
end
