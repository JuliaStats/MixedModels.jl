using GLM # bring r2 into scope
using LinearAlgebra
using MixedModels
using PooledArrays
using Random
using SparseArrays
using Suppressor
using Statistics
using StatsModels
using Tables
using Test
using TypedTables

using MixedModels: likelihoodratiotest

@isdefined(io) || const global io = IOBuffer()

include("modelcache.jl")

@testset "LMM from MixedModel" begin
    f = @formula(reaction ~ 1 + days + (1 | subj))
    d = MixedModels.dataset(:sleepstudy)
    @test MixedModel(f, d) isa LinearMixedModel
    @test MixedModel(f, d, Normal()) isa LinearMixedModel
    @test MixedModel(f, d, Normal(), IdentityLink()) isa LinearMixedModel
end

@testset "offset" begin
    let off = repeat([1], 180),
        slp = MixedModels.dataset(:sleepstudy),
        frm = @formula(reaction ~ 1 + (1 | subj))

        @test_throws ArgumentError fit(MixedModel, frm, slp; offset=off)
        @test_throws ArgumentError fit(
            MixedModel, frm, slp, Normal(), IdentityLink(); offset=off
        )
    end
end

@testset "Dyestuff" begin
    fm1 = only(models(:dyestuff))

    @test length(fm1.A) == 3
    @test size(fm1.reterms) == (1,)
    @test fm1.optsum.initial == ones(1)
    fm1.θ = ones(1)
    @test fm1.θ == ones(1)
    @test islinear(fm1)
    @test responsename(fm1) == "yield"
    @test meanresponse(fm1) ≈ 1527.5
    @test modelmatrix(fm1) == ones(30, 1)
    @test weights(fm1) == ones(30)

    @test_throws ArgumentError fit!(fm1)

    fm1.optsum.feval = -1
    @test_logs (:warn, "Model has not been fit") show(fm1)
    @test !isfitted(fm1)

    @test objective!(fm1, 0.713) ≈ 327.34216280954615

    show(io, BlockDescription(fm1))
    @test countlines(seekstart(io)) == 3
    output = String(take!(io))
    @test startswith(output, "rows:")

    refit!(fm1; progress=false)

    @test isfitted(fm1)
    @test :θ in propertynames(fm1)
    @test objective(fm1) ≈ 327.32705988112673 atol = 0.001
    @test fm1.θ ≈ [0.7525806540074477] atol = 1.e-5
    @test fm1.λ ≈ [LowerTriangular(reshape(fm1.θ, 1, :))]
    @test deviance(fm1) ≈ 327.32705988112673 atol = 0.001
    @test aic(fm1) ≈ 333.32705988112673 atol = 0.001
    @test bic(fm1) ≈ 337.5306520261132 atol = 0.001
    @test fixef(fm1) ≈ [1527.5]
    @test dispersion_parameter(fm1)
    @test first(first(fm1.σs)) ≈ 37.260343703061764 atol = 0.0001
    @test fm1.β ≈ [1527.5]
    @test dof(fm1) == 3
    @test nobs(fm1) == 30
    @test MixedModels.fixef!(zeros(1), fm1) ≈ [1527.5]
    @test coef(fm1) ≈ [1527.5]
    fm1β = fm1.βs
    @test fm1β isa NamedTuple
    @test isone(length(fm1β))
    @test first(values(fm1β)) ≈ 1527.5
    fm1σρ = fm1.σρs
    @test fm1σρ isa NamedTuple
    @test isone(length(fm1σρ))
    @test isone(length(getproperty(first(fm1σρ), :σ)))
    @test isempty(getproperty(first(fm1σρ), :ρ))
    @test fm1.σ == sdest(fm1)
    @test fm1.b == ranef(fm1)
    @test fm1.u == ranef(fm1; uscale=true)
    @test fm1.stderror == stderror(fm1)
    @test isone(length(fm1.pvalues))
    @test fm1.objective == objective(fm1)
    @test fm1.σ ≈ 49.51010035223816 atol = 1.e-5
    @test fm1.X == ones(30, 1)
    ds = MixedModels.dataset(:dyestuff)
    @test fm1.y == ds[:yield]
    @test response(fm1) == ds.yield
    @test cond(fm1) == ones(1)
    @test first(leverage(fm1)) ≈ 0.1565053420672158 rtol = 1.e-5
    @test sum(leverage(fm1)) ≈ 4.695160262016474 rtol = 1.e-5
    cm = coeftable(fm1)
    @test length(cm.rownms) == 1
    @test length(cm.colnms) == 4
    @test fnames(fm1) == (:batch,)
    @test response(fm1) == ds[:yield]
    rfu = ranef(fm1; uscale=true)
    rfb = ranef(fm1)
    @test abs(sum(only(rfu))) < 1.e-5
    cv = condVar(fm1)
    @test length(cv) == 1
    @test size(first(cv)) == (1, 1, 6)
    show(IOBuffer(), fm1.optsum)

    @test logdet(fm1) ≈ 8.06014611206176 atol = 0.001
    @test varest(fm1) ≈ 2451.2500368886936 atol = 0.001
    @test pwrss(fm1) ≈ 73537.50110666081 atol = 0.01 # this quantity is not precisely estimated
    @test stderror(fm1) ≈ [17.694552929494222] atol = 0.0001

    vc = VarCorr(fm1)
    show(io, vc)
    str = String(take!(io))
    @test startswith(str, "Variance components:")
    @test vc.s == sdest(fm1)

    refit!(fm1; REML=true, progress=false)
    @test objective(fm1) ≈ 319.6542768422576 atol = 0.0001
    @test_throws ArgumentError loglikelihood(fm1)
    @test dof_residual(fm1) ≥ 0

    print(io, fm1)
    @test startswith(String(take!(io)), "Linear mixed model fit by REML")

    vc = fm1.vcov
    @test isa(vc, Matrix{Float64})
    @test only(vc) ≈ 375.7167103872769 rtol = 1.e-3
    # since we're caching the fits, we should get it back to being correctly fitted
    # we also take this opportunity to test fitlog
    @testset "fitlog" begin
        refit!(fm1; REML=false, progress=false)
        fitlog = fm1.optsum.fitlog
        fitlogtbl = columntable(fm1.optsum)
        @test length(fitlogtbl) == 3
        @test keys(fitlogtbl) == (:iter, :objective, :θ)
        @test length(first(fitlogtbl)) > 15   # can't be sure of exact length
        @test first(fitlogtbl)[1:3] == 1:3
        @test last(fitlogtbl.objective) == fm1.optsum.fmin
        fitlogstackedtbl = columntable(fm1.optsum; stack=true)
        @test length(fitlogstackedtbl) == 4
        @test keys(fitlogstackedtbl) == (:iter, :objective, :par, :value)
        d, r = divrem(length(first(fitlogstackedtbl)), length(first(fitlogtbl)))
        @test iszero(r)
        @test d == length(first(fitlogtbl.θ))
    end
    @testset "profile" begin
        dspr01 = profile(only(models(:dyestuff)))
        sigma0row = only(filter(r -> r.p == :σ && iszero(r.ζ), dspr01.tbl))
        @test sigma0row.σ ≈ dspr01.m.σ
        @test sigma0row.β1 ≈ only(dspr01.m.β)
        @test sigma0row.θ1 ≈ only(dspr01.m.θ)
    end
end

@testset "Dyestuff2" begin
    fm = only(models(:dyestuff2))
    show(IOBuffer(), fm)
    @test fm.θ ≈ zeros(1)
    @test objective(fm) ≈ 162.87303665382575
    @test abs(only(first(std(fm)))) < 1.0e-9
    @test std(fm)[2] ≈ [3.6532313513746537]
    @test stderror(fm) ≈ [0.6669857396443264]
    @test coef(fm) ≈ [5.6656]
    @test logdet(fm) ≈ 0.0
    @test issingular(fm)
    #### modifies the model
    refit!(fm, float(MixedModels.dataset(:dyestuff)[:yield]); progress=false)
    @test objective(fm) ≈ 327.32705988112673 atol = 0.001
    refit!(fm, float(MixedModels.dataset(:dyestuff2)[:yield]); progress=false) # restore the model in the cache
    @testset "profile" begin   # tests a branch in profileσs! for σ estimate of zero
        dspr02 = profile(only(models(:dyestuff2)))
        sigma10row = only(filter(r -> r.p == :σ1 && iszero(r.ζ), dspr02.tbl))
        @test iszero(sigma10row.σ1)
        sigma1tbl = Table(filter(r -> r.p == :σ1, dspr02.tbl))
        @test all(≥(0), sigma1tbl.σ1)
    end
end

@testset "penicillin" begin
    fm = only(models(:penicillin))
    @test size(fm) == (144, 1, 30, 2)
    @test fm.optsum.initial == ones(2)

    @test objective(fm) ≈ 332.1883486700085 atol = 0.001
    @test coef(fm) ≈ [22.97222222222222] atol = 0.001
    @test fixef(fm) ≈ [22.97222222222222] atol = 0.001
    @test coef(fm)[1] ≈ mean(MixedModels.dataset(:penicillin).diameter)
    @test stderror(fm) ≈ [0.7446037806555799] atol = 0.0001
    @test fm.θ ≈ [1.5375939045981573, 3.219792193110907] atol = 0.001
    stdd = std(fm)
    @test only(first(stdd)) ≈ 0.845571948075415 atol = 0.0001
    @test only(stdd[2]) ≈ 1.770666460750787 atol = 0.0001
    @test only(last(stdd)) ≈ 0.549931906953287 atol = 0.0001
    @test varest(fm) ≈ 0.30242510228527864 atol = 0.0001
    @test logdet(fm) ≈ 95.74676552743833 atol = 0.005

    cv = condVar(fm)
    @test length(cv) == 2
    @test size(first(cv)) == (1, 1, 24)
    @test size(last(cv)) == (1, 1, 6)
    @test first(first(cv)) ≈ 0.07331356908917808 rtol = 1.e-4
    @test last(last(cv)) ≈ 0.04051591717427688 rtol = 1.e-4

    cv2 = condVar(fm, :sample)
    @test cv2 ≈ last(cv)
    rfu = ranef(fm; uscale=true)
    @test length(rfu) == 2
    @test first(first(rfu)) ≈ 0.5231574704291094 rtol = 1.e-4

    rfb = ranef(fm)
    @test length(rfb) == 2
    @test last(last(rfb)) ≈ -3.0018241391465703 rtol = 1.e-4

    show(io, BlockDescription(fm))
    @test countlines(seekstart(io)) == 4
    tokens = Set(split(String(take!(io)), r"\s+"))
    @test "Diag/Dense" in tokens
    @test "Diagonal" in tokens
end

@testset "pastes" begin
    fm = last(models(:pastes))
    @test size(fm) == (60, 1, 40, 2)
    @test fm.optsum.initial == ones(2)

    @test objective(fm) ≈ 247.9944658624955 atol = 0.001
    @test coef(fm) ≈ [60.0533333333333] atol = 0.001
    @test fixef(fm) ≈ [60.0533333333333] atol = 0.001
    @test stderror(fm) ≈ [0.6421355774401101] atol = 0.0001
    @test fm.θ ≈ [3.5269029347766856, 1.3299137410046242] atol = 0.001
    stdd = std(fm)
    @test only(first(stdd)) ≈ 2.90407793598792 atol = 0.001
    @test only(stdd[2]) ≈ 1.0950608007768226 atol = 0.0001
    @test only(last(stdd)) ≈ 0.8234073887751603 atol = 0.0001
    @test varest(fm) ≈ 0.677999727889528 atol = 0.0001
    @test logdet(fm) ≈ 101.03834542101686 atol = 0.001

    cv = condVar(fm)
    @test length(cv) == 2
    @test size(first(cv)) == (1, 1, 30)
    @test first(first(cv)) ≈ 1.1118647819999143 rtol = 1.e-4
    @test size(last(cv)) == (1, 1, 10)
    @test last(last(cv)) ≈ 0.850420001234007 rtol = 1.e-4

    show(io, BlockDescription(fm))
    @test countlines(seekstart(io)) == 4
    tokens = Set(split(String(take!(io)), r"\s+"))
    @test "Sparse" in tokens
    @test "Diagonal" in tokens

    lrt = likelihoodratiotest(models(:pastes)...)
    @test only(lrt.tests.pvalues) ≈ 0.5233767965780878 atol = 0.0001

    @testset "missing variables in formula" begin
        ae = ArgumentError(
            "The following formula variables are not present in the table: [:reaction, :joy, :subj]",
        )
        @test_throws(ae,
            fit(MixedModel, @formula(reaction ~ 1 + joy + (1 | subj)), dataset(:pastes)))
    end
end

@testset "InstEval" begin
    fm1 = models(:insteval)[2]              # at one time this was the fist of the :insteval models
    @test size(fm1) == (73421, 2, 4114, 3)
    @test fm1.optsum.initial == ones(3)

    spL = sparseL(fm1)
    @test size(spL) == (4114, 4114)
    @test 733090 < nnz(spL) < 733100

    @test objective(fm1) ≈ 237721.76877450474 atol = 0.001
    ftd1 = fitted(fm1)
    @test size(ftd1) == (73421,)
    @test ftd1 == predict(fm1)
    @test first(ftd1) ≈ 3.1787619026604945 atol = 0.0001
    resid1 = residuals(fm1)
    @test size(resid1) == (73421,)
    @test first(resid1) ≈ 1.8212380973395055 atol = 0.00001

    @testset "PCA" begin
        @test length(fm1.rePCA) == 3
        pca = MixedModels.PCA(fm1)
        @test length(pca) == 3
        @test :covcor in propertynames(first(pca))
        str = String(take!(io))
        show(io, first(pca); stddevs=true, variances=true)
        str = String(take!(io))
        @test !isempty(findall("Standard deviations:", str))
        @test !isempty(findall("Variances:", str))
    end

    show(io, BlockDescription(fm1))
    @test countlines(seekstart(io)) == 5
    tokens = Set(split(String(take!(io)), r"\s+"))
    @test "Sparse" in tokens
    @test "Sparse/Dense" in tokens
    @test "Diag/Dense" in tokens

    fm2 = first(models(:insteval))
    @test objective(fm2) ≈ 237585.5534151695 atol = 0.001
    @test size(fm2) == (73421, 28, 4100, 2)
end

@testset "sleep" begin
    fm = last(models(:sleepstudy))
    A11 = first(fm.A)
    @test isa(A11, UniformBlockDiagonal{Float64})
    @test isa(first(fm.L), UniformBlockDiagonal{Float64})
    @test size(A11) == (36, 36)
    a11 = view(A11.data, :, :, 1)
    @test a11 == [10.0 45.0; 45.0 285.0]
    @test size(A11.data, 3) == 18
    λ = only(fm.λ)
    b11 = LowerTriangular(view(first(fm.L).data, :, :, 1))
    @test b11 * b11' ≈ λ'a11 * λ + I rtol = 1e-5
    @test count(!iszero, Matrix(first(fm.L))) == 18 * 4
    @test rank(fm) == 2

    @test objective(fm) ≈ 1751.9393444636682
    @test fm.θ ≈ [0.9292297167514472, 0.01816466496782548, 0.22264601131030412] atol = 1.e-5
    @test pwrss(fm) ≈ 117889.27379003687 rtol = 1.e-5     # consider changing to log(pwrss) - this is too dependent even on AppleAccelerate vs OpenBLAS
    @test logdet(fm) ≈ 73.90350673367566 atol = 0.001
    @test stderror(fm) ≈ [6.632295312722272, 1.5022387911441102] atol = 0.0001
    @test coef(fm) ≈ [251.40510484848454, 10.467285959596126] atol = 1.e-5
    @test fixef(fm) ≈ [251.40510484848454, 10.467285959596126] atol = 1.e-5
    @test first(std(fm)) ≈ [23.78066438213187, 5.7168446983832775] atol = 0.01
    @test only(cond(fm)) ≈ 4.175266438717022 atol = 0.0001
    @test loglikelihood(fm) ≈ -875.9696722318341 atol = 1.e-5
    @test sum(leverage(fm)) ≈ 28.611653305323234 rtol = 1.e-5
    σs = fm.σs
    @test length(σs) == 1
    @test keys(σs) == (:subj,)
    @test length(σs.subj) == 2
    @test first(values(σs.subj)) ≈ 23.78066438213187 atol = 0.0001
    @test last(values(first(σs))) ≈ 5.7168446983832775 atol = 0.0001
    @test fm.corr ≈ [1.0 -0.13755599049585931; -0.13755599049585931 1.0] atol = 0.0001

    u3 = ranef(fm; uscale=true)
    @test length(u3) == 1
    @test size(first(u3)) == (2, 18)
    @test first(only(u3)) ≈ 3.030047743065841 atol = 0.001

    cv = condVar(fm)
    @test length(cv) == 1
    cv1 = only(cv)
    @test size(cv1) == (2, 2, 18)
    @test first(cv1) ≈ 140.96755256125914 rtol = 1.e-4
    @test last(cv1) ≈ 5.157794803497628 rtol = 1.e-4
    @test cv1[2] ≈ -20.604544204749537 rtol = 1.e-4

    cvt = condVartables(fm)
    @test length(cvt) == 1
    @test only(keys(cvt)) == :subj
    cvtsubj = cvt.subj
    @test only(cvt) === cvtsubj
    @test keys(cvtsubj) == (:subj, :σ, :ρ)
    @test Tables.istable(cvtsubj)
    @test first(cvtsubj.subj) == "S308"
    cvtsubjσ1 = first(cvtsubj.σ)
    @test all(==(cvtsubjσ1), cvtsubj.σ)
    @test first(cvtsubjσ1) ≈ 11.872975724781853 atol = 1.0e-4
    @test last(cvtsubjσ1) ≈ 2.271077894634534 atol = 1.0e-4
    cvtsubjρ = first(cvtsubj.ρ)
    @test all(==(cvtsubjρ), cvtsubj.ρ)
    @test only(cvtsubjρ) ≈ -0.7641373042040389 atol = 1.0e-4

    b3 = ranef(fm)
    @test length(b3) == 1
    @test size(only(b3)) == (2, 18)
    @test first(only(b3)) ≈ 2.8156104060324334 atol = 0.001

    b3tbl = raneftables(fm)
    @test length(b3tbl) == 1
    @test keys(b3tbl) == (:subj,)
    @test isa(b3tbl, NamedTuple)
    @test Tables.istable(only(b3tbl))

    @testset "PosDefException from constant response" begin
        slp = MixedModels.dataset(:sleepstudy)
        @test_throws ArgumentError(
            "The response is constant and thus model fitting has failed"
        ) refit!(fm, zero(slp.reaction); progress=false)
        refit!(fm, slp.reaction; progress=false)
    end

    simulate!(fm)  # to test one of the unscaledre methods
    # must restore state of fm as it is cached in the global fittedmodels
    slp = MixedModels.dataset(:sleepstudy)
    copyto!(fm.y, slp.reaction)
    updateL!(MixedModels.reevaluateAend!(fm))
    @test objective(fm) ≈ 1751.9393444636682 # check the model is properly restored

    fmnc = models(:sleepstudy)[2]
    @test size(fmnc) == (180, 2, 36, 1)
    @test fmnc.optsum.initial == ones(2)
    sigmas = fmnc.σs
    @test length(only(sigmas)) == 2
    @test first(only(sigmas)) ≈ 24.171361283849798 atol = 1e-4

    @testset "zerocorr PCA" begin
        @test length(fmnc.rePCA) == 1
        @test fmnc.rePCA.subj ≈ [0.5, 1.0]
        @test any(Ref(fmnc.PCA.subj.loadings) .≈ (I(2), I(2)[:, [2, 1]]))
        @test show(IOBuffer(), MixedModels.PCA(fmnc)) === nothing
    end

    @test deviance(fmnc) ≈ 1752.003255140962 atol = 0.001
    @test objective(fmnc) ≈ 1752.003255140962 atol = 0.001
    @test coef(fmnc) ≈ [251.4051048484854, 10.467285959595674]
    @test fixef(fmnc) ≈ [251.4051048484854, 10.467285959595674]
    @test stderror(fmnc) ≈ [6.707646513654387, 1.5193112497954953] atol = 0.001
    @test fmnc.θ ≈ [0.9458043022417869, 0.22692740996014607] atol = 0.0001
    @test first(std(fmnc)) ≈ [24.171269957611873, 5.79939919963132] atol = 0.0001
    @test last(std(fmnc)) ≈ [25.55613836753517] atol=0.0001
    @test logdet(fmnc) ≈ 74.4694698615524 atol = 0.001
    ρ = first(fmnc.σρs.subj.ρ)
    @test ρ === -0.0   # test that systematic zero correlations are returned as -0.0

    MixedModels.likelihoodratiotest(fm, fmnc)   # why is this stand-alone
    fmrs = fit(
        MixedModel, @formula(reaction ~ 1 + days + (0 + days | subj)), slp; progress=false
    )
    @test objective(fmrs) ≈ 1774.080315280526 rtol = 0.00001
    @test fmrs.θ ≈ [0.24353985601485326] rtol = 0.00001

    fm_ind = models(:sleepstudy)[3]
    @test objective(fm_ind) ≈ objective(fmnc)
    @test coef(fm_ind) ≈ coef(fmnc)
    @test fixef(fm_ind) ≈ fixef(fmnc)
    @test stderror(fm_ind) ≈ stderror(fmnc)
    @test fm_ind.θ ≈ fmnc.θ
    @test std(fm_ind) ≈ std(fmnc)
    @test logdet(fm_ind) ≈ logdet(fmnc)

    # combining [ReMat{T,S1}, ReMat{T,S2}] for S1 ≠ S2
    slpcat = (subj=slp.subj, days=PooledArray(string.(slp.days)), reaction=slp.reaction)
    fm_cat = fit(
        MixedModel,
        @formula(reaction ~ 1 + days + (1 | subj) + (0 + days | subj)),
        slpcat;
        progress=false,
    )
    @test fm_cat isa LinearMixedModel
    σρ = fm_cat.σρs
    @test σρ isa NamedTuple
    @test isone(length(σρ))
    @test first(keys(σρ)) == :subj
    @test keys(σρ.subj) == (:σ, :ρ)
    @test length(σρ.subj) == 2
    @test length(first(σρ.subj)) == 10
    @test length(σρ.subj.ρ) == 45
    # test that there's no correlation between the intercept and days columns
    ρs_intercept = σρ.subj.ρ[1 .+ cumsum(0:8)]
    @test all(iszero.(ρs_intercept))
    # amalgamate should set these to -0.0 to indicate structural zeros
    @test all(ρs_intercept .=== -0.0)

    # also works without explicitly dropped intercept
    fm_cat2 = fit(
        MixedModel,
        @formula(reaction ~ 1 + days + (1 | subj) + (days | subj)),
        slpcat;
        progress=false,
    )
    @test fm_cat2 isa LinearMixedModel
    σρ = fm_cat2.σρs
    @test σρ isa NamedTuple
    @test isone(length(σρ))
    @test first(keys(σρ)) == :subj
    @test keys(σρ.subj) == (:σ, :ρ)
    @test length(σρ.subj) == 2
    @test length(first(σρ.subj)) == 10
    @test length(σρ.subj.ρ) == 45
    # test that there's no correlation between the intercept and days columns
    ρs_intercept = σρ.subj.ρ[1 .+ cumsum(0:8)]
    @test all(iszero.(ρs_intercept))
    # amalgamate should set these to -0.0 to indicate structural zeros
    @test all(ρs_intercept .=== -0.0)

    @testset "diagonal λ in zerocorr" begin
        # explicit zerocorr
        fmzc = models(:sleepstudy)[2]
        λ = first(fmzc.reterms).λ
        @test λ isa Diagonal{Float64,Vector{Float64}}
        # implicit zerocorr via amalgamation
        fmnc = models(:sleepstudy)[3]
        λ = first(fmnc.reterms).λ
        @test λ isa Diagonal{Float64,Vector{Float64}}
    end

    @testset "disable amalgamation" begin
        fm_chunky = fit(MixedModel,
            @formula(reaction ~ 1 + days + (1 | subj) + (0 + days | subj)),
            dataset(:sleepstudy); amalgamate=false, progress=false)

        @test loglikelihood(fm_chunky) ≈ loglikelihood(models(:sleepstudy)[2])
        @test length(fm_chunky.reterms) == 2
        vc = sprint(show, VarCorr(fm_chunky))
        @test all(occursin(vc), ["subj", "subj.2"])
    end

    show(io, BlockDescription(first(models(:sleepstudy))))
    @test countlines(seekstart(io)) == 3
    @test "Diagonal" in Set(split(String(take!(io)), r"\s+"))

    show(io, BlockDescription(last(models(:sleepstudy))))
    @test countlines(seekstart(io)) == 3
    @test "BlkDiag" in Set(split(String(take!(io)), r"\s+"))

    @testset "optsumJSON" begin
        fm = refit!(last(models(:sleepstudy)); progress=false)
        # using a IOBuffer for saving JSON
        saveoptsum(seekstart(io), fm)
        m = LinearMixedModel(fm.formula, MixedModels.dataset(:sleepstudy))
        restoreoptsum!(m, seekstart(io))
        @test length(fm.optsum.fitlog) == length(m.optsum.fitlog)

        # try it out with an empty fitlog
        empty!(fm.optsum.fitlog)
        saveoptsum(seekstart(io), fm)
        restoreoptsum!(m, seekstart(io))
        # the restored fitlog always contains the initial and final values
        @test length(m.optsum.fitlog) == 2

        fm_mod = deepcopy(fm)
        fm_mod.optsum.fmin += 1
        saveoptsum(seekstart(io), fm_mod)
        @test_throws(
            ArgumentError(
                "model at final does not match stored fmin within atol=0.0, rtol=1.0e-8"
            ),
            restoreoptsum!(m, seekstart(io); atol=0.0, rtol=1e-8))
        restoreoptsum!(m, seekstart(io); atol=1)
        @test m.optsum.fmin - fm.optsum.fmin ≈ 1

        # using a temporary file for saving JSON
        fnm = first(mktemp())
        saveoptsum(fnm, fm)
        m = LinearMixedModel(fm.formula, MixedModels.dataset(:sleepstudy))
        restoreoptsum!(m, fnm)
        @test loglikelihood(fm) ≈ loglikelihood(m)
        @test bic(fm) ≈ bic(m)
        @test coef(fm) ≈ coef(m)

        # check restoreoptsum from older versions
        m = LinearMixedModel(
            @formula(reaction ~ 1 + days + (1 + days | subj)),
            MixedModels.dataset(:sleepstudy),
        )
        iob = IOBuffer(
            """
            {
                "initial":[1.0,0.0,1.0],
                "finitial":1784.642296192436,
                "ftol_rel":1.0e-12,
                "ftol_abs":1.0e-8,
                "xtol_rel":0.0,
                "xtol_abs":[1.0e-10,1.0e-10,1.0e-10],
                "initial_step":[0.75,1.0,0.75],
                "maxfeval":-1,
                "maxtime":-1.0,
                "feval":57,
                "final":[0.9292213195402981,0.01816837807519162,0.22264487477788353],
                "fmin":1751.9393444646712,
                "optimizer":"LN_BOBYQA",
                "returnvalue":"FTOL_REACHED",
                "nAGQ":1,
                "REML":false
            }
            """,
        )
        @test_logs(
            (:warn,
                r"optsum was saved with an older version of MixedModels.jl: consider resaving",
            ),
            restoreoptsum!(m, seekstart(iob)))
        @test loglikelihood(fm) ≈ loglikelihood(m)
        @test bic(fm) ≈ bic(m)
        @test coef(fm) ≈ coef(m)
        iob = IOBuffer(
            """
            {
                "initial":[1.0,0.0,1.0],
                "finitial":1784.642296192436,
                "ftol_rel":1.0e-12,
                "xtol_rel":0.0,
                "xtol_abs":[1.0e-10,1.0e-10,1.0e-10],
                "initial_step":[0.75,1.0,0.75],
                "maxfeval":-1,
                "maxtime":-1.0,
                "feval":57,
                "final":[0.9292213195402981,0.01816837807519162,0.22264487477788353],
                "fmin":1751.9393444646712,
                "optimizer":"LN_BOBYQA",
                "returnvalue":"FTOL_REACHED",
                "nAGQ":1,
                "REML":false,
                "sigma":null,
                "fitlog":[[[1.0,0.0,1.0],1784.642296192436]]
            }
            """,
        )
        @test_throws(ArgumentError("optsum names: [:ftol_abs] not found in io"),
            restoreoptsum!(m, seekstart(iob)))
        # note that this contains a fitlog from an older version!
        iob = IOBuffer(
            """
            {
                "initial":[1.0,0.0,1.0],
                "finitial":1784.642296192436,
                "ftol_rel":1.0e-12,
                "ftol_abs":1.0e-8,
                "xtol_rel":0.0,
                "xtol_abs":[1.0e-10,1.0e-10,1.0e-10],
                "rhobeg":1.0,
                "rhoend":1.0e-6,
                "xtol_zero_abs":0.001,
                "ftol_zero_abs":1.0e-5,
                "backend": "nlopt",
                "initial_step":[0.75,1.0,0.75],
                "maxfeval":-1,
                "maxtime":-1.0,
                "feval":57,
                "final":[0.9292213195402981,0.01816837807519162,0.22264487477788353],
                "fmin":1751.9393444646712,
                "optimizer":"LN_BOBYQA",
                "returnvalue":"FTOL_REACHED",
                "nAGQ":1,
                "REML":false,
                "sigma":null,
                "fitlog":[[[1.0,0.0,1.0],1784.642296192436]]
            }
            """,
        )
        @test_logs(
            (:warn,
                r"optsum was saved with an older version of MixedModels.jl: consider resaving",
            ),
            restoreoptsum!(m, seekstart(iob)))
        mktemp() do path, io
            m = deepcopy(last(models(:sleepstudy)))
            m.optsum.xtol_zero_abs = 0.5
            m.optsum.ftol_zero_abs = 0.5
            saveoptsum(io, m)
            m.optsum.xtol_zero_abs = 1.0
            m.optsum.ftol_zero_abs = 1.0
            @suppress restoreoptsum!(m, seekstart(io))
            @test m.optsum.xtol_zero_abs == 0.5
            @test m.optsum.ftol_zero_abs == 0.5
        end
    end

    @testset "profile" begin
        pr = @suppress profile(last(models(:sleepstudy)))
        tbl = pr.tbl
        @test length(tbl) >= 122
        ci = confint(pr)
        @test Tables.istable(ci)
        @test propertynames(ci) == (:par, :estimate, :lower, :upper)
        @test collect(ci.par) == [:β1, :β2, :σ, :σ1, :σ2]
        @test isapprox(
            ci.lower.values,
            [237.681, 7.359, 22.898, 14.381, 0.0];
            atol=1.e-3)
        @test isapprox(
            ci.upper.values,
            [265.130, 13.576, 28.858, 37.718, 8.753];
            atol=1.e-3)
        @test first(only(filter(r -> r.p == :σ && iszero(r.ζ), pr.tbl)).σ) ==
            last(models(:sleepstudy)).σ

        @testset "REML" begin
            m = refit!(deepcopy(last(models(:sleepstudy))); progress=false, REML=true)
            ci = @suppress confint(profile(m))
            @test all(splat(<), zip(ci.lower, ci.upper))
        end
    end
    @testset "confint" begin
        ci = confint(last(models(:sleepstudy)))
        @test Tables.istable(ci)
        @test isapprox(ci.lower.values, [238.4061184564825, 7.52295850741417]; atol=1.e-3)
    end

    @testset "Cook's Distance" begin
        lme4_cooks = [0.1270714, 0.1267805, 0.243096, 0.0002437091, 0.03145029, 0.2954052,
            0.04550505,
            0.3552723, 0.1984806, 0.4518805, 0.1683441, 0.02902698, 0.004232616,
            1.734029e-05,
            0.003816645, 0.00623334, 0.03219321, 0.05429389, 0.07319191, 0.06649928,
            0.007803994,
            0.001435875, 0.03886176, 0.01013682, 7.076106e-05, 0.02487801, 0.01538649,
            0.002299068,
            0.008366248, 0.08733211, 0.3043884, 0.0770035, 0.003193764, 0.000259058,
            0.00841487,
            0.00664586, 0.0894498, 0.007342141, 0.07721502, 0.00115366, 0.0476889,
            0.01107893,
            0.02342937, 0.04474152, 0.009826393, 0.02536012, 0.07157197, 8.781548e-08,
            0.1757661,
            0.01755979, 0.04308501, 0.04907289, 0.003603381, 0.02141832, 0.01529109,
            0.0002237688,
            1.055383, 0.01226195, 0.01122611, 0.7032865, 0.01801972, 0.008351314,
            0.009071886,
            1.922539e-05, 0.009401271, 0.01932602, 0.0001153177, 0.003751265, 0.02194446,
            4.78793e-09,
            0.02048001, 0.01981013, 0.04247507, 0.03844668, 0.007580713, 0.01639404,
            0.001973649,
            0.006080187, 0.0008513994, 0.08466273, 0.0878464, 0.2161317, 0.0467594,
            0.06665132,
            0.0006486227, 0.0009503809, 0.03397066, 0.1231246, 0.1946271, 0.2816787,
            0.008455713,
            0.02639438, 0.1743106, 0.00450064, 1.73262e-05, 0.01563701, 0.01998501,
            0.02539804,
            0.157366, 0.1206117, 0.002382807, 0.007197368, 0.009506474, 0.002782844,
            0.02747835,
            0.00986326, 0.008074464, 0.001298994, 0.03273043, 0.05191876, 0.005918988,
            0.0696993,
            0.05733613, 0.1038886, 0.0881868, 0.008494316, 0.159206, 0.03677518, 0.135499,
            0.06079108,
            0.003406159, 0.1399327, 0.001825492, 0.00191708, 0.01107303, 0.004549203,
            0.02109569,
            0.1587737, 0.002198379, 0.006746796, 0.3064917, 3.780973e-07, 0.02104387,
            0.04698987,
            0.02207251, 0.009852787, 0.0009590272, 1.506034e-05, 0.001194266, 0.003147009,
            0.01284797,
            1.315739e-05, 0.03073671, 0.00899036, 0.01262709, 0.002494427, 0.03239389,
            0.01698841,
            0.0002320865, 0.0135889, 0.02761053, 0.02916589, 0.04618232, 0.07875934,
            0.02248172,
            0.1308213, 0.04340534, 0.05379937, 0.0873526, 0.07648689, 0.03333461,
            0.01267992,
            0.004915966, 0.0003118122, 0.006997041, 0.01519545, 0.162238, 0.01767151,
            0.02365221,
            0.05187042, 1.31043e-07, 0.002747362, 0.003266733, 0.005808394, 0.03485179,
            0.003650455,
            0.0003004733, 1.535027e-05, 0.0168071, 1.510735e-05]
        model = refit!(first(models(:sleepstudy)); progress=false)
        @test all(zip(lme4_cooks, cooksdistance(model))) do (x, y)
            return isapprox(x, y; atol=1e-5)
        end
    end
end

@testset "d3" begin
    fm = only(models(:d3))
    @test pwrss(fm) ≈ 5.3047961973685445e6 rtol = 1.e-4
    @test objective(fm) ≈ 884957.5539373319 rtol = 1e-4
    @test coef(fm) ≈ [0.49912367745838365, 0.31130769168177186] atol = 1.e-4
    @test length(ranef(fm)) == 3
    @test sum(leverage(fm)) ≈ 8808.020656781464 rtol = 1.e-4

    show(io, BlockDescription(fm))
    tokens = Set(split(String(take!(io)), r"\s+"))
    @test "BlkDiag/Dense" in tokens
end

@testset "kb07" begin
    global io
    pca = last(models(:kb07)).PCA
    @test keys(pca) == (:subj, :item)
    show(io, models(:kb07)[2])
    @test sum(leverage(last(models(:kb07)))) ≈ 131.28414754217545 rtol = 7.e-3
    tokens = Set(split(String(take!(io)), r"\s+"))
    @test "Corr." in tokens
    @test "-0.89" in tokens
    @testset "profile" begin
        contrasts = Dict(:item => Grouping(), :subj => Grouping(),
            :prec => EffectsCoding(; base="maintain"),
            :spkr => EffectsCoding(), :load => EffectsCoding())
        kbf03 = @formula rt_trunc ~ 1 + prec + spkr + load + (1 + prec | item) + (1 | subj)
        kbpr03 = profile(
            fit(MixedModel, kbf03, MixedModels.dataset(:kb07); contrasts, progress=false)
        )
        @test length(Tables.columnnames(kbpr03.tbl)) == 15
        @test length(Tables.rows(kbpr03.tbl)) > 200
    end
end

@testset "oxide" begin
    # this model has an interesting structure with two diagonal blocks
    m = first(models(:oxide))
    @test isapprox(m.θ, [1.6892072390381156, 2.98500065754288]; atol=1e-3)
    # m = last(models(:oxide))
    # NB: this is a poorly defined fit
    # lme4 gives all sorts of convergence warnings for the different
    # optimizers and even quite different values
    # the overall estimates of the standard deviations are similar-ish
    # but the correlation structure seems particular unstable
    #θneldermead = [1.6454, 8.6373e-02, 8.2128e-05, 8.9552e-01, 1.2014, 2.9286]
    # two different BOBYQA implementations
    #θnlopt = [1.645, -0.221, 0.986, 0.895, 2.511, 1.169]
    #θminqa = [1.6455, -0.2430, 1.0160, 0.8955, 2.7054, 0.0898]
    # very loose tolerance for unstable fit
    # but this is a convenient test of rankUpdate!(::UniformBlockDiagonal)
    #    @test isapprox(m.θ, θnlopt; atol=5e-2)   # model doesn't make sense

    # @testset "profile" begin   # if the model fit doesn' make sense, profiling it makes even less sense
        # TODO: actually handle the case here so that it doesn't error and
        # create a separate test of the error handling code
    #     @test_logs((:error, "Exception occurred in profiling; aborting..."),
    #         @test_throws Exception profile(last(models(:oxide))))
    # end
end

@testset "Rank deficient" begin
    rng = MersenneTwister(0)
    x = rand(rng, 100)
    data = (x=x, x2=1.5 .* x, y=rand(rng, 100), z=repeat('A':'T', 5))
    model = @suppress fit(MixedModel, @formula(y ~ x + x2 + (1 | z)), data; progress=false)
    @test length(fixef(model)) == 2
    @test rank(model) == 2
    @test length(coef(model)) == 3
    ct = coeftable(model)
    @test ct.rownms == ["(Intercept)", "x", "x2"]
    @test length(fixefnames(model)) == 2
    @test coefnames(model) == ["(Intercept)", "x", "x2"]
    piv = model.feterm.piv
    r = model.feterm.rank
    @test coefnames(model)[piv][1:r] == fixefnames(model)
end

@testset "coeftable" begin
    ct = coeftable(only(models(:dyestuff)))
    @test [3, 4] == [ct.teststatcol, ct.pvalcol]
end

@testset "wts" begin
    # example from https://github.com/JuliaStats/MixedModels.jl/issues/194
    data = (
        a=[
            1.55945122,
            0.004391538,
            0.005554163,
            -0.173029772,
            4.586284429,
            0.259493671,
            -0.091735715,
            5.546487603,
            0.457734831,
            -0.030169602,
        ],
        b=[
            0.24520519,
            0.080624178,
            0.228083467,
            0.2471453,
            0.398994279,
            0.037213859,
            0.102144973,
            0.241380251,
            0.206570975,
            0.15980803,
        ],
        c=PooledArray(["H", "F", "K", "P", "P", "P", "D", "M", "I", "D"]),
        w1=[20, 40, 35, 12, 29, 25, 65, 105, 30, 75],
        w2=[
            0.04587156,
            0.091743119,
            0.080275229,
            0.027522936,
            0.066513761,
            0.05733945,
            0.149082569,
            0.240825688,
            0.068807339,
            0.172018349,
        ],
    )

    #= no need to fit yet another model without weights, but here are the reference values from lme4
    m1 = fit(MixedModel, @formula(a ~ 1 + b + (1|c)), data; progress=false)
    @test m1.θ ≈ [0.0]
    @test stderror(m1) ≈  [1.084912299335946, 4.966336338239706] atol = 1.e-4
    @test vcov(m1) ≈ [1.177034697250409 -4.80259802739442; -4.80259802739442 24.66449662452017] atol = 1.e-4
    =#

    m2 = fit(MixedModel, @formula(a ~ 1 + b + (1 | c)), data; wts=data.w1, progress=false)
    @test m2.θ ≈ [0.2951818091809752] atol = 1.e-4
    @test stderror(m2) ≈ [0.964016663994572, 3.6309691484830533] atol = 1.e-4
    @test vcov(m2) ≈
        [0.9293281284592235 -2.5575260810649962; -2.5575260810649962 13.18393695723575] atol =
        1.e-4
end

@testset "unifying ReMat eltypes" begin
    sleepstudy = MixedModels.dataset(:sleepstudy)

    re =
        LinearMixedModel(
            @formula(reaction ~ 1 + days + (1 | subj) + (days | subj)), sleepstudy
        ).reterms
    # make sure that the eltypes are still correct
    # otherwise this test isn't checking what it should be
    @test eltype(sleepstudy.days) == Int8
    @test eltype(sleepstudy.reaction) == Float64

    # use explicit typeof() and == is to remind us that things may break
    # if we change things and don't check their type implications now
    # that we're starting to support a non trivial type hierarchy
    @test typeof(re) == Vector{AbstractReMat{Float64}}
end

@testset "recovery from misscaling" begin
    model = fit(MixedModel,
        @formula(reaction ~ 1 + days + zerocorr(1 + fulldummy(days) | subj)),
        MixedModels.dataset(:sleepstudy);
        progress=false,
        contrasts=Dict(:days => HelmertCoding(),
            :subj => Grouping()))
    fm1 = MixedModels.unfit!(deepcopy(model))
    fm1.optsum.initial .*= 1e8
    @test_logs (:info, r"Initial objective evaluation failed") (
        :warn, r"Failure of the initial "
    ) fit!(fm1; progress=false)
    @test objective(fm1) ≈ objective(model) rtol = 0.1
    # it would be great to test the handling of PosDefException after the first iteration
    # but this is surprisingly hard to trigger in a reliable way across platforms
    # just because of the vagaries of floating point.
end

@testset "methods we don't define" begin
    m = first(models(:sleepstudy))
    for f in [r2, adjr2]
        @test_logs (:error,) @test_throws MethodError f(m)
    end
end
