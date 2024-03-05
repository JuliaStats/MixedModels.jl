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
    f = @formula(reaction ~ 1 + days + (1|subj))
    d = MixedModels.dataset(:sleepstudy)
    @test MixedModel(f, d) isa LinearMixedModel
    @test MixedModel(f, d, Normal()) isa LinearMixedModel
    @test MixedModel(f, d, Normal(), IdentityLink()) isa LinearMixedModel
end

@testset "offset" begin
    let off = repeat([1], 180),
        slp = MixedModels.dataset(:sleepstudy),
        frm = @formula(reaction ~ 1 + (1|subj))

        @test_throws ArgumentError fit(MixedModel, frm, slp; offset=off)
        @test_throws ArgumentError fit(MixedModel, frm, slp, Normal(), IdentityLink(); offset=off)
    end
end

@testset "Dyestuff" begin
    fm1 = only(models(:dyestuff))

    @test length(fm1.A) == 3
    @test size(fm1.reterms) == (1, )
    @test lowerbd(fm1) == zeros(1)
    @test fm1.lowerbd == zeros(1)
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

    @test objective(updateL!(setθ!(fm1, [0.713]))) ≈ 327.34216280955366

    show(io, BlockDescription(fm1))
    @test countlines(seekstart(io)) == 3
    output = String(take!(io))
    @test startswith(output, "rows:")

    refit!(fm1; progress=false)

    @test isfitted(fm1)
    @test :θ in propertynames(fm1)
    @test objective(fm1) ≈ 327.3270598811428 atol=0.001
    @test fm1.θ ≈ [0.752580] atol=1.e-5
    @test fm1.λ ≈ [LowerTriangular(reshape(fm1.θ, (1,1)))] atol=1.e-5
    @test deviance(fm1) ≈ 327.32705988 atol=0.001
    @test aic(fm1) ≈ 333.3270598811394 atol=0.001
    @test bic(fm1) ≈ 337.5306520261259 atol=0.001
    @test fixef(fm1) ≈ [1527.5]
    @test dispersion_parameter(fm1)
    @test first(first(fm1.σs)) ≈ 37.26034462135931 atol=0.0001
    @test fm1.β ≈ [1527.5]
    @test dof(fm1) == 3
    @test nobs(fm1) == 30
    @test MixedModels.fixef!(zeros(1),fm1) ≈ [1527.5]
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
    @test fm1.u == ranef(fm1, uscale=true)
    @test fm1.stderror == stderror(fm1)
    @test isone(length(fm1.pvalues))
    @test fm1.objective == objective(fm1)
    @test fm1.σ ≈ 49.510099986291145 atol=1.e-5
    @test fm1.X == ones(30,1)
    ds = MixedModels.dataset(:dyestuff)
    @test fm1.y == ds[:yield]
    @test response(fm1) == ds.yield
    @test cond(fm1) == ones(1)
    @test first(leverage(fm1)) ≈ 0.15650534392640486 rtol=1.e-5
    @test sum(leverage(fm1)) ≈ 4.695160317792145 rtol=1.e-5
    cm = coeftable(fm1)
    @test length(cm.rownms) == 1
    @test length(cm.colnms) == 4
    @test fnames(fm1) == (:batch,)
    @test response(fm1) == ds[:yield]
    rfu = ranef(fm1, uscale = true)
    rfb = ranef(fm1)
    @test abs(sum(rfu[1])) < 1.e-5
    cv = condVar(fm1)
    @test length(cv) == 1
    @test size(first(cv)) == (1, 1, 6)
    show(IOBuffer(), fm1.optsum)

    @test logdet(fm1) ≈ 8.06014522999825 atol=0.001
    @test varest(fm1) ≈ 2451.2501089607676 atol=0.001
    @test pwrss(fm1) ≈ 73537.50152584909 atol=0.01 # this quantity is not precisely estimated
    @test stderror(fm1) ≈ [17.69455188898009] atol=0.0001

    vc = VarCorr(fm1)
    show(io, vc)
    str = String(take!(io))
    @test startswith(str, "Variance components:")
    @test vc.s == sdest(fm1)

    refit!(fm1; REML=true, progress=false)
    @test objective(fm1) ≈ 319.65427684225216 atol=0.0001
    @test_throws ArgumentError loglikelihood(fm1)
    @test dof_residual(fm1) ≥ 0

    print(io, fm1)
    @test startswith(String(take!(io)), "Linear mixed model fit by REML")

    vc = fm1.vcov
    @test isa(vc, Matrix{Float64})
    @test only(vc) ≈ 375.7167775 rtol=1.e-3
    # since we're caching the fits, we should get it back to being correctly fitted
    # we also take this opportunity to test fitlog
    @testset "fitlog" begin
        thin = 1
        refit!(fm1; REML=false, progress=false, thin)
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
        thin = 2
        refit!(fm1; REML=false, progress=false, thin)
        @test length(fitlog) == (div(fm1.optsum.feval, thin) + 1) # for the initial value
        @test first(fitlog) == (fm1.optsum.initial, fm1.optsum.finitial)
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
    @test lowerbd(fm) == zeros(1)
    show(IOBuffer(), fm)
    @test fm.θ ≈ zeros(1)
    @test objective(fm) ≈ 162.87303665382575
    @test abs(std(fm)[1][1]) < 1.0e-9
    @test std(fm)[2] ≈ [3.653231351374652]
    @test stderror(fm) ≈ [0.6669857396443261]
    @test coef(fm) ≈ [5.6656]
    @test logdet(fm) ≈ 0.0
    @test issingular(fm)
    #### modifies the model
    refit!(fm, float(MixedModels.dataset(:dyestuff)[:yield]); progress=false)
    @test objective(fm) ≈ 327.3270598811428 atol=0.001
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
    @test lowerbd(fm) == zeros(2)

    @test objective(fm) ≈ 332.18834867227616 atol=0.001
    @test coef(fm) ≈ [22.97222222222222] atol=0.001
    @test fixef(fm) ≈ [22.97222222222222] atol=0.001
    @test coef(fm)[1] ≈ mean(MixedModels.dataset(:penicillin).diameter)
    @test stderror(fm) ≈ [0.7445960346851368] atol=0.0001
    @test fm.θ ≈ [1.5375772376554968, 3.219751321180035] atol=0.001
    @test first(std(fm)) ≈ [0.8455645948223015] atol=0.0001
    @test std(fm)[2] ≈ [1.770647779277388] atol=0.0001
    @test varest(fm) ≈ 0.3024263987592062 atol=0.0001
    @test logdet(fm) ≈ 95.74614821367786 atol=0.001

    cv = condVar(fm)
    @test length(cv) == 2
    @test size(first(cv)) == (1, 1, 24)
    @test size(last(cv)) == (1, 1, 6)
    @test first(first(cv)) ≈ 0.07331320237988301 rtol=1.e-4
    @test last(last(cv)) ≈ 0.04051547211287544 rtol=1.e-4

    cv2 = condVar(fm, :sample)
    @test cv2 ≈ last(cv)
    rfu = ranef(fm, uscale=true)
    @test length(rfu) == 2
    @test first(first(rfu)) ≈ 0.523162392717432 rtol=1.e-4

    rfb = ranef(fm)
    @test length(rfb) == 2
    @test last(last(rfb)) ≈ -3.001823834230942 rtol=1.e-4

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
    @test lowerbd(fm) == zeros(2)

    @test objective(fm) ≈ 247.99446586289676 atol=0.001
    @test coef(fm) ≈ [60.05333333333329] atol=0.001
    @test fixef(fm) ≈ [60.05333333333329] atol=0.001
    @test stderror(fm) ≈ [0.6421359883527029] atol=0.0001
    @test fm.θ ≈ [3.5268858714382905, 1.3299230213750168] atol=0.001
    @test first(std(fm)) ≈ [2.904069002535747] atol=0.001
    @test std(fm)[2] ≈ [1.095070371687089] atol=0.0001
    @test std(fm)[3] ≈ [0.8234088395243269] atol=0.0001
    @test varest(fm) ≈ 0.6780020742644107 atol=0.0001
    @test logdet(fm) ≈ 101.0381339953986 atol=0.001

    cv = condVar(fm)
    @test length(cv) == 2
    @test size(first(cv)) == (1, 1, 30)
    @test first(first(cv)) ≈ 1.111873335663485 rtol=1.e-4
    @test size(last(cv)) == (1, 1, 10)
    @test last(last(cv)) ≈ 0.850428770978789 rtol=1.e-4

    show(io, BlockDescription(fm))
    @test countlines(seekstart(io)) == 4
    tokens = Set(split(String(take!(io)), r"\s+"))
    @test "Sparse" in tokens
    @test "Diagonal" in tokens

    lrt = likelihoodratiotest(models(:pastes)...)
    @test length(lrt.deviance) == length(lrt.formulas) == length(lrt.models )== 2
    @test first(lrt.tests.pvalues) ≈ 0.5233767966395597 atol=0.0001

    @testset "missing variables in formula" begin
        ae = ArgumentError("The following formula variables are not present in the table: [:reaction, :joy, :subj]")
        @test_throws(ae,
                     fit(MixedModel, @formula(reaction ~ 1 + joy + (1|subj)), dataset(:pastes)))

    end
end

@testset "InstEval" begin
    fm1 = first(models(:insteval))
    @test size(fm1) == (73421, 2, 4114, 3)
    @test fm1.optsum.initial == ones(3)
    @test lowerbd(fm1) == zeros(3)

    spL = sparseL(fm1)
    @test size(spL) == (4114, 4114)
    @test 733090 < nnz(spL) < 733100

    @test objective(fm1) ≈ 237721.7687745563 atol=0.001
    ftd1 = fitted(fm1);
    @test size(ftd1) == (73421, )
    @test ftd1 == predict(fm1)
    @test first(ftd1) ≈ 3.17876 atol=0.0001
    resid1 = residuals(fm1);
    @test size(resid1) == (73421, )
    @test first(resid1) ≈ 1.82124 atol=0.00001

    @testset "PCA" begin
        @test length(fm1.rePCA) == 3
        pca = MixedModels.PCA(fm1)
        @test length(pca) == 3
        @test :covcor in propertynames(first(pca))
        str = String(take!(io))
        show(io, first(pca), stddevs=true, variances=true)
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

    fm2 = last(models(:insteval))
    @test objective(fm2) ≈ 237585.5534151694 atol=0.001
    @test size(fm2) == (73421, 28, 4100, 2)
end

@testset "sleep" begin
    fm = last(models(:sleepstudy))
    @test lowerbd(fm) == [0.0, -Inf, 0.0]
    A11 = first(fm.A)
    @test isa(A11, UniformBlockDiagonal{Float64})
    @test isa(first(fm.L), UniformBlockDiagonal{Float64})
    @test size(A11) == (36, 36)
    a11 = view(A11.data, :, :, 1)
    @test a11 == [10. 45.; 45. 285.]
    @test size(A11.data, 3) == 18
    λ = first(fm.λ)
    b11 = LowerTriangular(view(first(fm.L).data, :, :, 1))
    @test b11 * b11' ≈ λ'a11*λ + I rtol=1e-5
    @test count(!iszero, Matrix(first(fm.L))) == 18 * 4
    @test rank(fm) == 2

    @test objective(fm) ≈ 1751.9393444647046
    @test fm.θ ≈ [0.929221307, 0.01816838, 0.22264487096] atol=1.e-6
    @test pwrss(fm) ≈ 117889.46144025437
    @test logdet(fm) ≈ 73.90322021999222 atol=0.001
    @test stderror(fm) ≈ [6.632257721914501, 1.5022354739749826] atol=0.0001
    @test coef(fm) ≈ [251.40510484848477,10.4672859595959]
    @test fixef(fm) ≈  [251.40510484848477,10.4672859595959]
    @test std(fm)[1] ≈ [23.780468100188497, 5.716827903196682] atol=0.01
    @test logdet(fm) ≈ 73.90337187545992 atol=0.001
    @test cond(fm) ≈ [4.175251] atol=0.0001
    @test loglikelihood(fm) ≈ -875.9696722323523
    @test sum(leverage(fm)) ≈ 28.611525700136877 rtol=1.e-5
    σs = fm.σs
    @test length(σs) == 1
    @test keys(σs) == (:subj,)
    @test length(σs.subj) == 2
    @test first(values(σs.subj)) ≈ 23.7804686 atol=0.0001
    @test last(values(first(σs))) ≈ 5.7168278 atol=0.0001
    @test fm.corr ≈ [1.0 -0.1375451787621904; -0.1375451787621904 1.0] atol=0.0001

    u3 = ranef(fm, uscale=true)
    @test length(u3) == 1
    @test size(first(u3)) == (2, 18)
    @test first(u3)[1, 1] ≈ 3.030300122575336 atol=0.001

    cv = condVar(fm)
    @test length(cv) == 1
    @test size(first(cv)) == (2, 2, 18)
    @test first(first(cv)) ≈ 140.96612241084617 rtol=1.e-4
    @test last(last(cv)) ≈ 5.157750215432247 rtol=1.e-4
    @test first(cv)[2] ≈ -20.60428045516186 rtol=1.e-4

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
    @test first(cvtsubjσ1) ≈ 11.87291549750297 atol=1.0e-4
    @test last(cvtsubjσ1) ≈ 2.271068078114843 atol=1.0e-4
    cvtsubjρ = first(cvtsubj.ρ)
    @test all(==(cvtsubjρ), cvtsubj.ρ)
    @test only(cvtsubjρ) ≈ -0.7641347018831385 atol=1.0e-4

    b3 = ranef(fm)
    @test length(b3) == 1
    @test size(first(b3)) == (2, 18)
    @test first(first(b3)) ≈ 2.815819441982976 atol=0.001

    b3tbl = raneftables(fm)
    @test length(b3tbl) == 1
    @test keys(b3tbl) == (:subj,)
    @test isa(b3tbl, NamedTuple)
    @test Tables.istable(only(b3tbl))

    @testset "PosDefException from constant response" begin
        slp = MixedModels.dataset(:sleepstudy)
        @test_throws ArgumentError("The response is constant and thus model fitting has failed") refit!(fm, zero(slp.reaction); progress=false)
        refit!(fm, slp.reaction; progress=false)
    end

    simulate!(fm)  # to test one of the unscaledre methods
    # must restore state of fm as it is cached in the global fittedmodels
    slp = MixedModels.dataset(:sleepstudy)
    copyto!(fm.y, slp.reaction)
    updateL!(MixedModels.reevaluateAend!(fm))
    @test objective(fm) ≈ 1751.9393444647046 # check the model is properly restored

    fmnc = models(:sleepstudy)[2]
    @test size(fmnc) == (180,2,36,1)
    @test fmnc.optsum.initial == ones(2)
    @test lowerbd(fmnc) == zeros(2)
    sigmas = fmnc.σs
    @test length(only(sigmas)) == 2
    @test first(only(sigmas)) ≈ 24.171449484676224 atol=1e-4

    @testset "zerocorr PCA" begin
        @test length(fmnc.rePCA) == 1
        @test fmnc.rePCA.subj ≈ [0.5, 1.0]
        @test any(Ref(fmnc.PCA.subj.loadings) .≈ (I(2), I(2)[:, [2,1]]))
        @test show(IOBuffer(), MixedModels.PCA(fmnc)) === nothing
    end

    @test deviance(fmnc) ≈ 1752.0032551398835 atol=0.001
    @test objective(fmnc) ≈ 1752.0032551398835 atol=0.001
    @test coef(fmnc) ≈ [251.40510484848585, 10.467285959595715]
    @test fixef(fmnc) ≈ [251.40510484848477, 10.467285959595715]
    @test stderror(fmnc) ≈ [6.707710260366577, 1.5193083237479683] atol=0.001
    @test fmnc.θ ≈ [0.9458106880922268, 0.22692826607677266] atol=0.0001
    @test first(std(fmnc)) ≈ [24.171449463289047, 5.799379721123582]
    @test last(std(fmnc)) ≈ [25.556130034081047]
    @test logdet(fmnc) ≈ 74.46952585564611 atol=0.001
    ρ = first(fmnc.σρs.subj.ρ)
    @test ρ === -0.0   # test that systematic zero correlations are returned as -0.0

    MixedModels.likelihoodratiotest(fm, fmnc)
    fmrs = fit(MixedModel, @formula(reaction ~ 1+days + (0+days|subj)), slp; progress=false);
    @test objective(fmrs) ≈ 1774.080315280528 rtol=0.00001
    @test fmrs.θ ≈ [0.24353985679033105] rtol=0.00001

    fm_ind = models(:sleepstudy)[3]
    @test objective(fm_ind) ≈ objective(fmnc)
    @test coef(fm_ind) ≈ coef(fmnc)
    @test fixef(fm_ind) ≈ fixef(fmnc)
    @test stderror(fm_ind) ≈ stderror(fmnc)
    @test fm_ind.θ ≈ fmnc.θ
    @test std(fm_ind) ≈ std(fmnc)
    @test logdet(fm_ind) ≈ logdet(fmnc)

    # combining [ReMat{T,S1}, ReMat{T,S2}] for S1 ≠ S2
    slpcat = (subj = slp.subj, days = PooledArray(string.(slp.days)), reaction = slp.reaction)
    fm_cat = fit(MixedModel, @formula(reaction ~ 1+days+(1|subj)+(0+days|subj)),slpcat; progress=false)
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
    fm_cat2 = fit(MixedModel, @formula(reaction ~ 1+days+(1|subj)+(days|subj)),slpcat; progress=false)
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
        @test λ isa Diagonal{Float64, Vector{Float64}}
        # implicit zerocorr via amalgamation
        fmnc = models(:sleepstudy)[3]
        λ = first(fmnc.reterms).λ
        @test λ isa Diagonal{Float64, Vector{Float64}}
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
        fm = last(models(:sleepstudy))
        # using a IOBuffer for saving JSON
        saveoptsum(seekstart(io), fm)
        m = LinearMixedModel(fm.formula, MixedModels.dataset(:sleepstudy))
        restoreoptsum!(m, seekstart(io))
        @test loglikelihood(fm) ≈ loglikelihood(m)
        @test bic(fm) ≈ bic(m)
        @test coef(fm) ≈ coef(m)

        fm_mod = deepcopy(fm)
        fm_mod.optsum.fmin += 1
        saveoptsum(seekstart(io), fm_mod)
        @test_throws(ArgumentError("model m at final does not give stored fmin"),
                     restoreoptsum!(m, seekstart(io)))
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
    end

    @testset "profile" begin
        pr = profile(last(models(:sleepstudy)))
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
        @test first(only(filter(r -> r.p == :σ && iszero(r.ζ), pr.tbl)).σ) == last(models(:sleepstudy)).σ
    end
    @testset "confint" begin
        ci = confint(last(models(:sleepstudy)))
        @test Tables.istable(ci)
        @test isapprox(ci.lower.values, [238.4061184564825, 7.52295850741417]; atol=1.e-3)
    end
end

@testset "d3" begin
    fm = only(models(:d3))
    @test pwrss(fm) ≈ 5.30480294295329e6 rtol=1.e-4
    @test objective(fm) ≈ 884957.5540213 rtol = 1e-4
    @test coef(fm) ≈ [0.4991229873, 0.31130780953] atol = 1.e-4
    @test length(ranef(fm)) == 3
    @test sum(leverage(fm)) ≈ 8808.00706143011 rtol = 1.e-4

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
        contrasts = Dict(:item => Grouping(), :subj => Grouping(), :prec => EffectsCoding(; base="maintain"),
        :spkr => EffectsCoding(), :load => EffectsCoding())
        kbf03 = @formula rt_trunc ~ 1+prec+spkr+load+(1+prec|item)+(1|subj)
        kbpr03 = profile(fit(MixedModel, kbf03, MixedModels.dataset(:kb07); contrasts, thin=1, progress=false))
        @test length(Tables.columnnames(kbpr03.tbl)) == 15
        @test length(Tables.rows(kbpr03.tbl)) > 200
    end
end

@testset "oxide" begin
    # this model has an interesting structure with two diagonal blocks
    m = first(models(:oxide))
    @test isapprox(m.θ, [1.689182746, 2.98504262]; atol=1e-3)
    m = last(models(:oxide))
    # NB: this is a poorly defined fit
    # lme4 gives all sorts of convergence warnings for the different
    # optimizers and even quite different values
    # the overall estimates of the standard deviations are similar-ish
    # but the correlation structure seems particular unstable
    θneldermead = [1.6454, 8.6373e-02, 8.2128e-05, 8.9552e-01, 1.2014, 2.9286]
    # two different BOBYQA implementations
    θnlopt = [1.645, -0.221, 0.986, 0.895, 2.511, 1.169]
    θminqa = [1.6455, -0.2430, 1.0160, 0.8955, 2.7054, 0.0898]
    # very loose tolerance for unstable fit
    # but this is a convenient test of rankUpdate!(::UniformBlockDiagonal)
    @test isapprox(m.θ, θnlopt; atol=5e-2)
end

@testset "Rank deficient" begin
    rng = MersenneTwister(0);
    x = rand(rng, 100);
    data = (x = x, x2 = 1.5 .* x, y = rand(rng, 100), z = repeat('A':'T', 5))
    model = @suppress fit(MixedModel, @formula(y ~ x + x2 + (1|z)), data; progress=false)
    @test length(fixef(model)) == 2
    @test rank(model) == 2
    @test length(coef(model)) == 3
    ct = coeftable(model)
    @test ct.rownms ==  ["(Intercept)", "x", "x2"]
    @test length(fixefnames(model)) == 2
    @test coefnames(model) == ["(Intercept)", "x", "x2"]
    piv = model.feterm.piv
    r = model.feterm.rank
    @test coefnames(model)[piv][1:r] == fixefnames(model)
end

@testset "coeftable" begin
    ct = coeftable(only(models(:dyestuff)));
    @test [3,4] == [ct.teststatcol, ct.pvalcol]
end

@testset "wts" begin
    # example from https://github.com/JuliaStats/MixedModels.jl/issues/194
    data = (
        a = [1.55945122,0.004391538,0.005554163,-0.173029772,4.586284429,0.259493671,-0.091735715,5.546487603,0.457734831,-0.030169602],
        b = [0.24520519,0.080624178,0.228083467,0.2471453,0.398994279,0.037213859,0.102144973,0.241380251,0.206570975,0.15980803],
        c = PooledArray(["H","F","K","P","P","P","D","M","I","D"]),
        w1 = [20,40,35,12,29,25,65,105,30,75],
        w2 = [0.04587156,0.091743119,0.080275229,0.027522936,0.066513761,0.05733945,0.149082569,0.240825688,0.068807339,0.172018349],
    )

    #= no need to fit yet another model without weights, but here are the reference values from lme4
    m1 = fit(MixedModel, @formula(a ~ 1 + b + (1|c)), data; progress=false)
    @test m1.θ ≈ [0.0]
    @test stderror(m1) ≈  [1.084912, 4.966336] atol = 1.e-4
    @test vcov(m1) ≈ [1.177035 -4.802598; -4.802598 24.664497] atol = 1.e-4
    =#

    m2 = fit(MixedModel, @formula(a ~ 1 + b + (1|c)), data; wts = data.w1, progress=false)
    @test m2.θ ≈ [0.295181729258352]  atol = 1.e-4
    @test stderror(m2) ≈  [0.9640167, 3.6309696] atol = 1.e-4
    @test vcov(m2) ≈ [0.9293282 -2.557527; -2.5575267 13.183940] atol = 1.e-4
end

@testset "unifying ReMat eltypes" begin
    sleepstudy = MixedModels.dataset(:sleepstudy)

    re = LinearMixedModel(@formula(reaction ~ 1 + days + (1|subj) + (days|subj)), sleepstudy).reterms
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
                @formula(reaction ~ 1 + days + zerocorr(1+fulldummy(days)|subj)),
                MixedModels.dataset(:sleepstudy);
                progress=false,
                contrasts=Dict(:days => HelmertCoding(),
                               :subj => Grouping()))
    fm1 = MixedModels.unfit!(deepcopy(model))
    fm1.optsum.initial .*= 1e8
    @test_logs (:info, r"Initial objective evaluation failed") (:warn, r"Failure of the initial ") fit!(fm1; progress=false)
    @test objective(fm1) ≈ objective(model) rtol=0.1
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
