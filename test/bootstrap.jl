using DataFrames
using LinearAlgebra
using MixedModels
using Random
using Statistics
using StableRNGs
using Statistics
using Suppressor
using Tables
using Test

using MixedModels: dataset, MixedModelBootstrap

include("modelcache.jl")

function quickboot(m, n=2)
    return parametricbootstrap(MersenneTwister(42), n, m;
                               progress=false, use_threads=false,
                               optsum_overrides=(;ftol_rel=1e-8))
end


@testset "simulate!(::MixedModel)" begin
    @testset "LMM" begin
        ds = dataset(:dyestuff)
        fm = only(models(:dyestuff))
        # # just in case the fit was modified in a previous test
        # refit!(fm, vec(float.(ds.yield)))
        resp₀ = copy(response(fm))
        # type conversion of ints to floats
        simulate!(StableRNG(1234321), fm, β=[1], σ=1)
        refit!(fm, resp₀; progress=false)
        refit!(simulate!(StableRNG(1234321), fm); progress=false)
        @test deviance(fm) ≈ 322.6582 atol=0.001
        refit!(fm, float(ds.yield), progress=false)
        # Global/implicit RNG method
        Random.seed!(1234321)
        refit!(simulate!(fm); progress=false)
        # just make sure this worked, don't check fit
        # (because the RNG can change between Julia versions)
        @test response(fm) ≠ resp₀
        simulate!(fm, θ = fm.θ)
        @test_throws DimensionMismatch refit!(fm, zeros(29); progress=false)
        # restore the original state
        refit!(fm, vec(float.(ds.yield)); progress=false)

        @testset "zerocorr" begin
            fmzc = models(:sleepstudy)[2]
            @test length(simulate(fmzc)) == length(response(fmzc))
        end
    end
    @testset "Poisson" begin
        center(v::AbstractVector) = v .- (sum(v) / length(v))
        grouseticks = DataFrame(dataset(:grouseticks))
        grouseticks.ch = center(grouseticks.height)
        gm4 = fit(MixedModel, only(gfms[:grouseticks]), grouseticks, Poisson(), fast=true, progress=false)  # fails in pirls! with fast=false
        gm4sim = refit!(simulate!(StableRNG(42), deepcopy(gm4)); progress=false)
        @test isapprox(gm4.β, gm4sim.β; atol=norm(stderror(gm4)))
    end

    @testset "Binomial" begin
        cbpp = dataset(:cbpp)
        gm2 = fit(MixedModel, first(gfms[:cbpp]), cbpp, Binomial(), wts=float(cbpp.hsz), progress=false)
        gm2sim = refit!(simulate!(StableRNG(42), deepcopy(gm2)); fast=true, progress=false)
        @test isapprox(gm2.β, gm2sim.β; atol=norm(stderror(gm2)))
    end
    @testset "_rand with dispersion" begin
        @test_throws ArgumentError MixedModels._rand(StableRNG(42), Normal(), 1, 1, 1)
        @test_throws ArgumentError MixedModels._rand(StableRNG(42), Gamma(), 1, 1, 1)
        @test_throws ArgumentError MixedModels._rand(StableRNG(42), InverseGaussian(), 1, 1, 1)
    end
end

@testset "bootstrap" begin
    fm = only(models(:dyestuff))
    # two implicit tests
    # 1. type conversion of ints to floats
    # 2. test method for default RNG
    @test_logs((:warn, r"hide_progress"),
                parametricbootstrap(1, fm, β=[1], σ=1, hide_progress=true))

    bsamp = parametricbootstrap(MersenneTwister(1234321), 100, fm;
                                use_threads=false, progress=false)
    @test isa(propertynames(bsamp), Vector{Symbol})
    @test length(bsamp.objective) == 100
    @test keys(first(bsamp.fits)) == (:objective, :σ, :β, :se, :θ)
    @test isa(bsamp.σs, Vector{<:NamedTuple})
    @test length(bsamp.σs) == 100
    allpars = DataFrame(bsamp.allpars)
    @test isa(allpars, DataFrame)

    @testset "optsum_overrides" begin
        bsamp2 = parametricbootstrap(MersenneTwister(1234321), 100, fm;
                                    use_threads=false, progress=false,
                                    optsum_overrides=(;ftol_rel=1e-8))
        # for such a simple, small model setting the function value
        # tolerance has little effect until we do something extreme
        @test bsamp.objective ≈ bsamp2.objective
        bsamp2 = parametricbootstrap(MersenneTwister(1234321), 100, fm;
                                    use_threads=false, progress=false,
                                    optsum_overrides=(;ftol_rel=1.0))
        @test !(bsamp.objective ≈ bsamp2.objective)
    end
    cov = shortestcovint(shuffle(1.:100.))
    # there is no unique shortest coverage interval here, but the left-most one
    # is currently returned, so we take that. If this behavior changes, then
    # we'll have to change the test
    @test first(cov) == 1.
    @test last(cov) == 95.

    coefp = DataFrame(bsamp.coefpvalues)

    @test isa(coefp, DataFrame)
    @test coefp.iter == 1:100
    @test only(unique(coefp.coefname)) == Symbol("(Intercept)")
    @test propertynames(coefp) == [:iter, :coefname, :β, :se, :z, :p]

    @testset "threaded bootstrap" begin
        @test_logs (:warn, r"use_threads is deprecated") parametricbootstrap(MersenneTwister(1234321), 1, fm;
                                                                             use_threads=true, progress=false)
    end

    @testset "zerocorr + Base.length + ftype" begin
        fmzc = models(:sleepstudy)[2]
        pbzc = parametricbootstrap(MersenneTwister(42), 5, fmzc, Float16;
                                   progress=false)
        @test length(pbzc) == 5
        @test Tables.istable(shortestcovint(pbzc))
        @test typeof(pbzc) == MixedModelBootstrap{Float16}
    end

    @testset "zerocorr + not zerocorr" begin
        form_zc_not = @formula(rt_trunc ~ 1 + spkr * prec * load +
                                         (1 + spkr + prec + load | subj) +
                                 zerocorr(1 + spkr + prec + load | item))
        fmzcnot = fit(MixedModel, form_zc_not, dataset(:kb07); progress=false)
        pbzcnot = parametricbootstrap(MersenneTwister(42), 2, fmzcnot, Float16;
                                      progress=false)
    end

    @testset "vcat" begin
        sleep = quickboot(last(models(:sleepstudy)))
        zc1 = quickboot(models(:sleepstudy)[2])
        zc2 = quickboot(models(:sleepstudy)[3])

        @test_throws ArgumentError vcat(sleep, zc1)
        @test_throws ArgumentError reduce(vcat, [sleep, zc1])
        # these are the same model even if the formulae
        # are expressed differently
        @test length(vcat(zc1, zc2)) == 4
        @test length(reduce(vcat, [zc1, zc2])) == 4
    end

    @testset "save and restore replicates" begin
        io = IOBuffer()
        m0 = first(models(:sleepstudy))
        m1 = last(models(:sleepstudy))
        pb0 = quickboot(m0)
        pb1 = quickboot(m1)
        savereplicates(io, pb1)
        @test isa(pb0.tbl, Table)
        @test isa(pb1.tbl, Table)  # create tbl here to check it doesn't modify pb1
        @test ncol(DataFrame(pb1.β)) == 3

        # wrong model
        @test_throws ArgumentError restorereplicates(seekstart(io), m0)
        # need to specify an eltype!
        @test_throws MethodError restorereplicates(seekstart(io), m1, MixedModelBootstrap)

        # make sure exact and approximate equality work
        @test pb1 == pb1
        @test pb1 == restorereplicates(seekstart(io), m1)
        @test pb1 ≈ restorereplicates(seekstart(io), m1)
        @test pb1 ≈ pb1
        @test pb1 ≈ restorereplicates(seekstart(io), m1, Float64)
        @test restorereplicates(seekstart(io), m1, Float32) ≈ restorereplicates(seekstart(io), m1, Float32)
        # too much precision is lost
        f16 = restorereplicates(seekstart(io), m1, Float16)
        @test !isapprox(pb1, f16)
        @test isapprox(pb1, f16; atol=eps(Float16))
        @test isapprox(pb1, f16; rtol=0.0001)


        # two paths, one destination
        @test restorereplicates(seekstart(io), m1, MixedModelBootstrap{Float16}) == restorereplicates(seekstart(io), m1, Float16)
        # changing eltype breaks exact equality
        @test pb1 != restorereplicates(seekstart(io), m1, Float32)

        # test that we don't need the model to be fit when restoring
        @test pb1 == restorereplicates(seekstart(io), MixedModels.unfit!(deepcopy(m1)))

        @test pb1 ≈ restorereplicates(seekstart(io), m1, Float16) rtol=1
    end

    @testset "Bernoulli simulate! and GLMM bootstrap" begin
        contra = dataset(:contra)
        # need a model with fast=false to test that we only
        # copy the optimizer constraints for θ and not β
        gm0 = fit(MixedModel, first(gfms[:contra]), contra, Bernoulli(), fast=false, progress=false)
        bs = parametricbootstrap(StableRNG(42), 100, gm0; progress=false)
        # make sure we're not copying
        @test length(bs.lowerbd) == length(gm0.θ)
        bsci = filter!(:type => ==("β"), DataFrame(shortestcovint(bs)))
        ciwidth = 2 .* stderror(gm0)
        waldci = DataFrame(coef=fixefnames(gm0),
                           lower=fixef(gm0) .- ciwidth,
                           upper=fixef(gm0) .+ ciwidth)

        # coarse tolerances because we're not doing many bootstrap samples
        @test all(isapprox.(bsci.lower, waldci.lower; atol=0.5))
        @test all(isapprox.(bsci.upper, waldci.upper; atol=0.5))

        σbar = mean(MixedModels.tidyσs(bs)) do x; x.σ end
        @test σbar ≈ 0.56 atol=0.1
        apar = filter!(row -> row.type == "σ", DataFrame(MixedModels.allpars(bs)))
        @test !("Residual" in apar.names)
        @test mean(apar.value) ≈ σbar

        # can't specify dispersion for families without that parameter
        @test_throws ArgumentError parametricbootstrap(StableRNG(42), 100, gm0;
                                                       σ=2, progress=false)
        @test sum(issingular(bs)) == 0
    end

    @testset "Rank deficient" begin
        rng = MersenneTwister(0);
        x = rand(rng, 100);
        data = (x = x, x2 = 1.5 .* x, y = rand(rng, [0,1], 100), z = repeat('A':'T', 5))
        @testset "$family" for family in [Normal(), Bernoulli()]
            model = @suppress fit(MixedModel, @formula(y ~ x + x2 + (1|z)), data, family; progress=false)
            boot = quickboot(model, 10)

            dropped_idx = model.feterm.piv[end]
            dropped_coef = coefnames(model)[dropped_idx]
            @test all(boot.β) do nt
                # if we're the dropped coef, then we must be -0.0
                # need isequal because of -0.0
                return nt.coefname != dropped_coef || isequal(nt.β, -0.0)
            end

            yc = simulate(StableRNG(1), model; β=coef(model))
            yf = simulate(StableRNG(1), model; β=fixef(model))
            @test all(x -> isapprox(x...), zip(yc, yf))
        end

        @testset "partial crossing" begin
            id = lpad.(string.(1:40), 2, "0")
            B = ["b0", "b1", "b2"]
            C = ["c0", "c1", "c2", "c3", "c4"]
            df = DataFrame(reshape(collect(Iterators.product(B, C, id)), :), [:b, :c, :id])
            df[!, :y] .= 0
            filter!(df) do row
                b = last(row.b)
                c = last(row.c)
                return b != c
            end

            m = LinearMixedModel(@formula(y ~ 1 + b * c + (1|id)), df)
            β = 1:rank(m)
            σ = 1
            simulate!(StableRNG(628), m; β, σ)
            fit!(m)

            boot = parametricbootstrap(StableRNG(271828), 1000,  m);
            bootci = DataFrame(shortestcovint(boot))
            filter!(:group => ismissing, bootci)
            select!(bootci, :names => disallowmissing => :coef, :lower, :upper)
            transform!(bootci, [:lower, :upper] => ByRow(middle) => :mean)

            @test all(x -> isapprox(x[1], x[2]; atol=0.1),  zip(coef(m), bootci.mean))
        end
    end
end

@testset "show and summary" begin
    fmzc = models(:sleepstudy)[2]
    level = 0.68
    pb = parametricbootstrap(MersenneTwister(42), 500, fmzc; progress=false)
    pr = profile(fmzc)
    @test startswith(sprint(show, MIME("text/plain"), pr),
                     "MixedModelProfile -- Table with 9 columns and 151 rows:")
    @test startswith(sprint(show, MIME("text/plain"), pb),
                     "MixedModelBootstrap with 500 samples\n     parameter  min        q25       median    mean      q75       max\n  ")

    df = DataFrame(pr)
    @test nrow(df) == 151
    @test propertynames(df) == collect(propertynames(pr.tbl))

    @testset "CI method comparison" begin
        level = 0.68
        ci_boot_equaltail = confint(pb; level, method=:equaltail)
        ci_boot_shortest = confint(pb; level, method=:shortest)
        @test_throws ArgumentError confint(pb; level, method=:other)
        ci_wald = confint(fmzc; level)
        ci_prof = confint(pr; level)
        @test first(ci_boot_shortest.lower, 2) ≈ first(ci_prof.lower, 2) atol=0.5
        @test first(ci_boot_equaltail.lower, 2) ≈ first(ci_prof.lower, 2) atol=0.5
        @test first(ci_prof.lower, 2) ≈ first(ci_wald.lower, 2) atol=0.1
    end
end
