using DataFrames
using LinearAlgebra
using MixedModels
using Random
using SparseArrays
using StatsModels
using Test

using MixedModels: dataset, levels, modelcols, nlevs

const LMM = LinearMixedModel

@testset "scalarReMat" begin
    ds = dataset("dyestuff")
    f1 = @formula(yield ~ 1 + (1|batch))
    y1, Xs1 = modelcols(apply_schema(f1, schema(ds), LMM), ds)
    sf = Xs1[2]
    psts = dataset("pastes")
    f2 = @formula(strength ~ 1 + (1|batch/cask))
    y2, Xs2 = modelcols(apply_schema(f2, schema(psts), LMM), psts)
    sf1 = Xs2[2]
    sf2 = Xs2[3]

    @testset "size" begin
        @test size(sf) == (30, 6)
        @test size(sf,1) == 30
        @test size(sf,2) == 6
        @test size(sf,3) == 1
        @test size(sf1) == (60, 10)
        @test size(sf2) == (60, 30)
    end

    @testset "utilities" begin
        @test levels(sf) == string.('A':'F')
        @test refpool(sf) == levels(sf)
        @test refarray(sf) == repeat(1:6, inner=5)
        @test refvalue(sf, 3) == "C"
        @test nlevs(sf) == 6
        @test eltype(sf) == Float64
        @test sparse(sf) == sparse(1:30, sf.refs, ones(30))
        fsf = Matrix(sf)
        @test size(fsf) == (30, 6)
        @test count(!iszero, fsf) == 30
        @test sort!(unique(fsf)) == [0.0, 1.0]
        @test cond(sf) == 1.0
        @test MixedModels.nθ(sf) == 1
        @test MixedModels.getθ(sf) == ones(1)
        @test MixedModels.getθ!(Vector{Float64}(undef, 1), sf) == ones(1)
        @test lowerbd(sf) == zeros(1)
        @test MixedModels.getθ(setθ!(sf, [0.5])) == [0.5]
        MixedModels.unscaledre!(Vector{Float64}(undef, 30), sf)
        @test_throws DimensionMismatch MixedModels.getθ!(Float64[], sf)
        @test_throws DimensionMismatch setθ!(sf, ones(2))
    end

    @testset "products" begin
        @test ones(30, 1)'sf == fill(5.0, (1, 6))
        @test mul!(Array{Float64}(undef, (size(sf1, 2), size(sf2, 2))), sf1', sf2) == Array(sf1'sf2)

        crp = sf'sf
        @test isa(crp, Diagonal{Float64})
        crp1 = copy(crp)
        @test crp1 == crp
        @test crp[2,6] == 0
        @test crp[6,6] == 5
        @test size(crp) == (6,6)
        @test crp.diag == fill(5.,6)
        rhs = y1'sf
        @test rhs == reshape([7525.0,7640.0,7820.0,7490.0,8000.0,7350.0], (1, 6))
        @test ldiv!(crp, copy(rhs)') == [1505.,1528.,1564.,1498.,1600.,1470.]

        @test isa(sf1'sf1, Diagonal{Float64})
        @test isa(sf2'sf2, Diagonal{Float64})
        @test isa(sf2'sf1,SparseMatrixCSC{Float64})

        @test MixedModels.lmulΛ!(sf', ones(6)) == fill(0.5, 6)
        @test MixedModels.rmulΛ!(ones(6, 6), sf) == fill(0.5, (6, 6))
    end

    @testset "reweight!" begin
        wts = rand(MersenneTwister(1234321), size(sf, 1))
        @test isapprox(vec(MixedModels.reweight!(sf, wts).wtz), wts)
    end
end

@testset "RandomEffectsTerm" begin
    slp = dataset("sleepstudy")
    contrasts =  Dict{Symbol,Any}()

    @testset "Detect same variable as blocking and experimental" begin
        f = @formula(reaction ~ 1 + (1 + subj|subj))
        @test_throws ArgumentError apply_schema(f, schema(f, slp, contrasts), LMM)
    end

    @testset "Detect both blocking and experimental variables" begin
        # note that U is not in the fixed effects because we want to make square
        # that we're detecting all the variables in the random effects
        f = @formula(reaction ~ 1 + (1 + days|subj))
        form = apply_schema(f, schema(f, slp, contrasts), LMM)
        @test StatsModels.termvars(form.rhs) == [:days, :subj]
    end

    @testset "Runtime construction of random effects terms" begin
        # operator precedence and basic terms:
        @test term(:a) | term(:b) isa RandomEffectsTerm
        @test term(1) + term(:a) | term(:b) isa RandomEffectsTerm
        @test term(1) + term(:a) + term(:a) & term(:c) | term(:b) isa RandomEffectsTerm

        # sleep study data:
        r, d, s, one = term.((:reaction, :days, :subj, 1))

        f1 = @formula(reaction ~ 1 + (1 + days | subj))
        f2 = r ~ one + (one + d | s)
        @test f2.rhs[end] isa RandomEffectsTerm
        ff1 = apply_schema(f1, schema(slp), LMM)
        ff2 = apply_schema(f2, schema(slp), LMM)
        # equality of RE terms not defined so check that they generate same modelcols
        @test modelcols(ff1.rhs[end], slp) == modelcols(ff2.rhs[end], slp)

        m1 = fit(LMM, f1, slp; progress=false)
        m2 = fit(LMM, f2, slp; progress=false)
        @test all(m1.λ .== m2.λ)

        @test StatsModels.terms(f2.rhs[end]) == [one, d, s]
        @test StatsModels.termvars(f2.rhs[end]) == [d.sym, s.sym]
    end

    @testset "Runtime construction of ZeroCorr" begin
        r, d, s, one = term.((:reaction, :days, :subj, 1))

        f1 = @formula(reaction ~ 1 + zerocorr(1 + days | subj))
        f2 = r ~ one + zerocorr(one + d | s)
        @test f2.rhs[end] isa MixedModels.ZeroCorr
        ff1 = apply_schema(f1, schema(slp), LMM)
        ff2 = apply_schema(f2, schema(slp), LMM)
        # equality of RE terms not defined so check that they generate same modelcols
        mc1 = modelcols(ff1.rhs[end], slp)
        mc2 = modelcols(ff2.rhs[end], slp)

        # test that zerocorr actually worked
        @test mc1.inds == mc2.inds == [1, 4]

        m1 = fit(LMM, f1, slp; progress=false)
        m2 = fit(LMM, f2, slp; progress=false)
        @test all(m1.λ .== m2.λ)

        @test StatsModels.terms(f2.rhs[end]) == [one, d, s]
        @test StatsModels.termvars(f2.rhs[end]) == [d.sym, s.sym]
    end

    @testset "ZeroCorr delegation" begin
        r, d, s, one = term.((:reaction, :days, :subj, 1))

        f = @formula(0 ~ 1 + days).rhs
        zc = zerocorr(one + d | s)
        @test f == zc.lhs
        @test zc.rhs.sym == :subj
    end

    @testset "Amalgamation of ZeroCorr with other terms" begin
        f = @formula(reaction ~ 1 + days + (1|subj) + zerocorr(days|subj))
        m = LMM(f, dataset(:sleepstudy), contrasts = Dict(:days => DummyCoding()))
        re = only(m.reterms)
        @test length(re.cnames) == length(unique(re.cnames)) == 10
    end
end

@testset "random effects term syntax" begin

    dat = (y = rand(18),
           g = string.(repeat('a':'f', inner=3)),
           f = string.(repeat('A':'C', outer=6)))

    @testset "fulldummy" begin
        @test_throws ArgumentError fulldummy(1)

        f = @formula(y ~ 1 + fulldummy(f))
        f1 = apply_schema(f, schema(dat))
        @test typeof(last(f1.rhs.terms)) <: FunctionTerm{typeof(fulldummy)}
        @test_throws ArgumentError modelcols(f1, dat)

        f2 = apply_schema(f, schema(dat), MixedModel)
        @test typeof(last(f2.rhs.terms)) <: CategoricalTerm{<:StatsModels.FullDummyCoding}
        @test modelcols(f2.rhs, dat)[1:3, :] == [1 1 0 0
                                                 1 0 1 0
                                                 1 0 0 1]

        # implicit intercept
        ff = apply_schema(@formula(y ~ 1 + (f | g)), schema(dat), MixedModel)
        rem = modelcols(last(ff.rhs), dat)
        @test size(rem) == (18, 18)
        @test rem[1:3, 1:4] == [1 0 0 0
                                1 1 0 0
                                1 0 1 0]

        # explicit intercept
        ff = apply_schema(@formula(y ~ 1 + (1+f | g)), schema(dat), MixedModel)
        rem = modelcols(last(ff.rhs), dat)
        @test size(rem) == (18, 18)
        @test rem[1:3, 1:4] == [1 0 0 0
                                1 1 0 0
                                1 0 1 0]

        # explicit intercept + full dummy
        ff = apply_schema(@formula(y ~ 1 + (1+fulldummy(f) | g)), schema(dat), MixedModel)
        rem = modelcols(last(ff.rhs), dat)
        @test size(rem) == (18, 24)
        @test rem[1:3, 1:4] == [1 1 0 0
                                1 0 1 0
                                1 0 0 1]

        # explicit dropped intercept (implicit full dummy)
        ff = apply_schema(@formula(y ~ 1 + (0+f | g)), schema(dat), MixedModel)
        rem = modelcols(last(ff.rhs), dat)
        @test size(rem) == (18, 18)
        @test rem[1:3, 1:4] == [1 0 0 0
                                0 1 0 0
                                0 0 1 0]
    end

    @testset "nesting" begin
        ff = apply_schema(@formula(y ~ 1 + (1|g/f)), schema(dat), MixedModel)
        @test modelcols(last(ff.rhs), dat) == float(Matrix(I, 18, 18))

        # in fixed effects:
        d2 = (a = rand(20), b = repeat([:X, :Y], outer=10), c = repeat([:S,:T],outer=10))
        f2 = apply_schema(@formula(0 ~ 1 + b/a), schema(d2), MixedModel)
        @test modelcols(f2.rhs, d2) == [ones(20) d2.b .== :Y (d2.b .== :X).*d2.a (d2.b .== :Y).*d2.a]
        @test coefnames(f2.rhs) == ["(Intercept)", "b: Y", "b: X & a", "b: Y & a"]

        # check promotion
        f3 = apply_schema(@formula(0 ~ 0 + b/a), schema(d2), MixedModel)
        @test modelcols(f3.rhs, d2) == [d2.b .== :X d2.b .== :Y (d2.b .== :X).*d2.a (d2.b .== :Y).*d2.a]
        @test coefnames(f3.rhs) == ["b: X", "b: Y", "b: X & a", "b: Y & a"]

        # errors for continuous grouping
        @test_throws ArgumentError apply_schema(@formula(0 ~ 1 + a/b), schema(d2), MixedModel)

        # errors for too much nesting
        @test_throws ArgumentError apply_schema(@formula(0 ~ 1 + b/c/a), schema(d2), MixedModel)

        # fitted model to test amalgamate and fnames, and equivalence with other formulations
        psts = dataset("pastes")
        m = fit(MixedModel, @formula(strength ~ 1 + (1|batch/cask)), psts; progress=false)
        m2 = fit(MixedModel, @formula(strength ~ 1 + (1|batch) + (1|batch&cask)), psts; progress=false)
        m2r = fit(MixedModel, term(:strength) ~ term(1) + (term(1)|term(:batch)) + (term(1)|term(:batch)&term(:cask)), psts; progress=false)

        @test fnames(m) == fnames(m2) == fnames(m2r) == (Symbol("batch & cask"), :batch)
        @test coefnames(first(m.reterms)) == ["(Intercept)"]
        @test m.λ == m2.λ == m2r.λ
        @test deviance(m) == deviance(m2) == deviance(m2r)
    end

    @testset "multiple terms with same grouping" begin
        dat = MixedModels.dataset(:kb07)
        sch = schema(dat)
        f1 = @formula(rt_trunc ~ 1 + (1 + prec + load | spkr))
        ff1 = apply_schema(f1, sch, MixedModel)

        retrm = last(ff1.rhs)
        @test last(retrm.lhs.terms).contrasts.contrasts isa DummyCoding

        f2 = @formula(rt_trunc ~ 1 + (1 + prec | spkr) + (0 + load | spkr))
        ff2 = apply_schema(f2, sch, MixedModel)

        retrm2 = last(ff2.rhs)
        @test last(retrm2.lhs.terms).contrasts.contrasts isa DummyCoding
    end
end
