using BlockArrays, DataFrames, LinearAlgebra, MixedModels, NamedArrays
using Random, RData, SparseArrays, Statistics, Tables, Test

if !@isdefined(dat) || !isa(dat, Dict{Symbol, DataFrame})
    const dat = Dict(Symbol(k) => v for (k, v) in
        load(joinpath(dirname(pathof(MixedModels)), "..", "test", "dat.rda")))
end

const LMM = LinearMixedModel

@testset "Dyestuff" begin
    fm1 = LMM(@formula(Y ~ 1 + (1|G)), dat[:Dyestuff])

    @test BlockArrays.nblocks(fm1.A) == (3, 3)
    @test size(fm1.reterms) == (1, )
    @test BlockArrays.nblocks(fm1.L) == (3, 3)
    @test lowerbd(fm1) == zeros(1)
    @test fm1.lowerbd == zeros(1)
    @test fm1.θ == ones(1)
    fm1.θ = ones(1)
    @test fm1.θ == ones(1)
#    @test_warn "Model has not been fit" show(fm1)

    @test objective(updateL!(setθ!(fm1, [0.713]))) ≈ 327.34216280955366
    MixedModels.describeblocks(IOBuffer(), fm1)

    fit!(fm1);
    @test :θ in propertynames(fm1)
    @test objective(fm1) ≈ 327.3270598811428 atol=0.001
    @test fm1.θ ≈ [0.752580] atol=1.e-5
    @test fm1.λ ≈ [LowerTriangular(reshape(fm1.θ, (1,1)))] atol=1.e-5
    @test deviance(fm1) ≈ 327.32705988 atol=0.001
    @test aic(fm1) ≈ 333.3270598811394 atol=0.001
    @test bic(fm1) ≈ 337.5306520261259 atol=0.001
    @test fixef(fm1) ≈ [1527.5]
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
    @test fm1.y == dat[:Dyestuff][!, :Y]
    @test cond(fm1) == ones(1)
    cm = coeftable(fm1)
    @test length(cm.rownms) == 1
    @test length(cm.colnms) == 4
    @test fnames(fm1) == (:G,)
    @test response(fm1) == dat[:Dyestuff][!, :Y]
    rfu = ranef(fm1, uscale = true)
    rfb = ranef(fm1)
    @test abs(sum(rfu[1])) < 1.e-5
    cv = condVar(fm1)
    @test length(cv) == 1
    @test size(first(cv)) == (1, 1, 6)
    show(IOBuffer(), fm1.optsum)

    @test logdet(fm1) ≈ 8.06014522999825 atol=0.001
    @test varest(fm1) ≈ 2451.2501089607676 atol=0.001
    @test pwrss(fm1) ≈ 73537.49947885796 atol=0.001
    @test stderror(fm1) ≈ [17.69455188898009] atol=0.0001

    vc = VarCorr(fm1)
    show(IOBuffer(), vc)
    @test vc.s == sdest(fm1)

    fit!(fm1, REML=true)
    @test objective(fm1) ≈ 319.65427684225216 atol=0.0001
    @test_throws ArgumentError loglikelihood(fm1)
    @test dof_residual(fm1) ≥ 0
    print(IOBuffer(), fm1)
end

@testset "Dyestuff2" begin
    fm = fit(MixedModel, @formula(Y ~ 1 + (1 | G)), dat[:Dyestuff2])
    @test lowerbd(fm) == zeros(1)
    show(IOBuffer(), fm)
    @test fm.θ ≈ zeros(1)
    @test objective(fm) ≈ 162.87303665382575
    @test abs(std(fm)[1][1]) < 1.0e-9
    @test std(fm)[2] ≈ [3.653231351374652]
    @test stderror(fm) ≈ [0.6669857396443261]
    @test coef(fm) ≈ [5.6656]
    @test logdet(fm) ≈ 0.0
    refit!(fm, dat[:Dyestuff][!, :Y])
    @test objective(fm) ≈ 327.3270598811428 atol=0.001
end

@testset "penicillin" begin
    fm = LMM(@formula(Y ~ 1 + (1 | G) + (1 | H)), dat[:Penicillin]);
    @test size(fm) == (144, 1, 30, 2)
    @test fm.θ == ones(2)
    @test lowerbd(fm) == zeros(2)

    fit!(fm)

    @test objective(fm) ≈ 332.18834867227616 atol=0.001
    @test coef(fm) ≈ [22.97222222222222] atol=0.001
    @test fixef(fm) ≈ [22.97222222222222] atol=0.001
    @test coef(fm)[1] ≈ mean(dat[:Penicillin][!, :Y])
    @test stderror(fm) ≈ [0.7445960346851368] atol=0.0001
    @test fm.θ ≈ [1.5375772376554968, 3.219751321180035] atol=0.001
    @test first(std(fm)) ≈ [0.8455645948223015] atol=0.0001
    @test std(fm)[2] ≈ [1.770647779277388] atol=0.0001
    @test varest(fm) ≈ 0.3024263987592062 atol=0.0001
    @test logdet(fm) ≈ 95.74614821367786 atol=0.001
    rfu = ranef(fm, uscale=true)
    rfb = ranef(fm)
    @test length(rfb) == 2
end

@testset "pastes" begin
    fm = LMM(@formula(Y ~ (1 | G) + (1 | H)), dat[:Pastes])
    @test size(fm) == (60, 1, 40, 2)
    @test fm.θ == ones(2)
    @test lowerbd(fm) == zeros(2)

    fit!(fm);

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
end

@testset "InstEval" begin
    fm1 = LMM(@formula(Y ~ 1 + A + (1 | G) + (1 | H) + (1 | I)), dat[:InstEval])
    @test size(fm1) == (73421, 2, 4114, 3)
    @test fm1.θ == ones(3)
    @test lowerbd(fm1) == zeros(3)

    fit!(fm1);

    @test objective(fm1) ≈ 237721.7687745563 atol=0.001
    ftd1 = fitted(fm1);
    @test size(ftd1) == (73421, )
    @test ftd1 == predict(fm1)
    @test first(ftd1) ≈ 3.17876 atol=0.0001
    resid1 = residuals(fm1);
    @test size(resid1) == (73421, )
    @test first(resid1) ≈ 1.82124 atol=0.00001
    @test length(fm1.rePCA) == 3

    fm2 = fit!(LinearMixedModel(@formula(Y ~ 1 + A*I + (1 | G) + (1 | H)), dat[:InstEval]))
    @test objective(fm2) ≈ 237585.5534151694 atol=0.001
    @test size(fm2) == (73421, 28, 4100, 2)
end

@testset "sleep" begin
    fm = LinearMixedModel(@formula(Y ~ 1 + U + (1 + U | G)), dat[:sleepstudy]);
    @test lowerbd(fm) == [0.0, -Inf, 0.0]
    A11 = fm.A[Block(1,1)]
    @test isa(A11, UniformBlockDiagonal{Float64})
    @test isa(fm.L[Block(1, 1)], UniformBlockDiagonal{Float64})
    @test size(A11) == (36, 36)
    @test A11.facevec[1] == [10. 45.; 45. 285.]
    @test length(A11.facevec) == 18
    updateL!(fm);
    b11 = LowerTriangular(fm.L[Block(1, 1)].facevec[1])
    @test b11 * b11' == fm.A[Block(1, 1)].facevec[1] + I
    @test count(!iszero, Matrix(fm.L[Block(1, 1)])) == 18 * 4
    @test rank(fm) == 2

    fit!(fm)

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
    σs = fm.σs
    @test length(σs) == 1
    @test keys(σs) == (:G,)
    @test length(σs.G) == 2
    @test first(values(σs.G)) ≈ 23.780468626896745 atol=0.0001
    @test last(values(first(σs))) ≈ 5.716827808126002 atol=0.0001
    show(IOBuffer(), fm)

    u3 = ranef(fm, uscale=true)
    @test length(u3) == 1
    @test size(first(u3)) == (2, 18)
    @test first(u3)[1, 1] ≈ 3.030300122575336 atol=0.001
    u3n = first(ranef(fm, uscale=true, named=true))
    @test u3n isa NamedArrays.NamedArray

    b3 = ranef(fm)
    @test length(b3) == 1
    @test size(first(b3)) == (2, 18)
    @test first(first(b3)) ≈ 2.815819441982976 atol=0.001

    simulate!(fm)  # to test one of the unscaledre methods

    fmnc = zerocorr!(LinearMixedModel(@formula(Y ~ 1 + U + (1+U|G)), dat[:sleepstudy]));
    @test size(fmnc) == (180,2,36,1)
    @test fmnc.θ == ones(2)
    @test lowerbd(fmnc) == zeros(2)

    fit!(fmnc)

    @test deviance(fmnc) ≈ 1752.0032551398835 atol=0.001
    @test objective(fmnc) ≈ 1752.0032551398835 atol=0.001
    @test coef(fmnc) ≈ [251.40510484848585, 10.467285959595715]
    @test fixef(fmnc) ≈ [251.40510484848477, 10.467285959595715]
    @test stderror(fmnc) ≈ [6.707710260366577, 1.5193083237479683] atol=0.001
    @test fmnc.θ ≈ [0.9458106880922268, 0.22692826607677266] atol=0.0001
    @test first(std(fmnc)) ≈ [24.171449463289047, 5.799379721123582]
    @test last(std(fmnc)) ≈ [25.556130034081047]
    @test logdet(fmnc) ≈ 74.46952585564611 atol=0.001
    ρ = first(fmnc.σρs.G.ρ)
    @test ρ === -0.0   # test that systematic zero correlations are returned as -0.0

    fmnc2 = LinearMixedModel(@formula(Y ~ 1 + U + zerocorr(1+U|G)),
                             dat[:sleepstudy])
    @test size(fmnc2) == (180,2,36,1)
    @test fmnc2.θ == ones(2)
    @test lowerbd(fmnc2) == zeros(2)

    fit!(fmnc2)

    @test deviance(fmnc2) ≈ 1752.0032551398835 atol=0.001
    @test objective(fmnc2) ≈ 1752.0032551398835 atol=0.001
    @test coef(fmnc2) ≈ [251.40510484848585, 10.467285959595715]
    @test fixef(fmnc2) ≈ [251.40510484848477, 10.467285959595715]
    @test stderror(fmnc2) ≈ [6.707710260366577, 1.5193083237479683] atol=0.001
    @test fmnc2.θ ≈ [0.9458106880922268, 0.22692826607677266] atol=0.0001
    @test first(std(fmnc2)) ≈ [24.171449463289047, 5.799379721123582]
    @test last(std(fmnc2)) ≈ [25.556130034081047]
    @test logdet(fmnc2) ≈ 74.46952585564611 atol=0.001


#    MixedModels.lrt(fm, fmnc)

    MixedModels.likelihoodratiotest(fm, fmnc)

    fmrs = LinearMixedModel(@formula(Y ~ 1 + U + (0 + U|G)), dat[:sleepstudy]);
    fit!(fmrs)
    @test objective(fmrs) ≈ 1774.080315280528 rtol=0.00001
    @test fmrs.θ ≈ [0.24353985679033105] rtol=0.00001

    fm_ind = LinearMixedModel(@formula(Y ~ 1 + U + (1|G) + (0 + U|G)), dat[:sleepstudy]);
    fit!(fm_ind)
    @test objective(fm_ind) ≈ objective(fmnc)
    @test coef(fm_ind) ≈ coef(fmnc)
    @test fixef(fm_ind) ≈ fixef(fmnc)
    @test stderror(fm_ind) ≈ stderror(fmnc)
    @test fm_ind.θ ≈ fmnc.θ
    @test std(fm_ind) ≈ std(fmnc)
    @test logdet(fm_ind) ≈ logdet(fmnc)

    # combining [ReMat{T,S1}, ReMat{T,S2}] for S1 ≠ S2
    slpcat = categorical!(deepcopy(dat[:sleepstudy]), [:U])
    fm_cat = LMM(@formula(Y ~ 1 + U + (1|G) + (0 + U|G)), slpcat)
    @test fm_cat isa LMM
    σρ = fit!(fm_cat).σρs
    @test σρ isa NamedTuple
    @test isone(length(σρ))
    @test first(keys(σρ)) == :G
    @test keys(σρ.G) == (:σ, :ρ)
    @test length(σρ.G) == 2
    @test length(first(σρ.G)) == 11
    @test length(σρ.G.ρ) == 55
    @test iszero(σρ.G.ρ[46])
    @test σρ.G.ρ[46] === -0.0
end

@testset "d3" begin
    fm = updateL!(LMM(@formula(Y ~ 1 + U + (1+U|G) + (1+U|H) + (1+U|I)), dat[:d3]));
    @test pwrss(fm) ≈ 5.1261847180180885e6 rtol = 1e-6
    @test objective(fm) ≈ 901641.2930413672 rtol = 1e-6
    fit!(fm)
    @test objective(fm) ≈ 884957.5540213 rtol = 1e-6
    @test coef(fm) ≈ [0.4991229873, 0.31130780953] atol = 1.e-4
    bv = ranef(fm)
    @test length(bv) == 3
end

@testset "simulate!" begin
    fm = fit(MixedModel, @formula(Y ~ 1 + (1 | G)), dat[:Dyestuff])
    refit!(simulate!(Random.MersenneTwister(1234321), fm))
    @test deviance(fm) ≈ 339.0218639362958 atol=0.001
    refit!(fm, dat[:Dyestuff][!, :Y])
    Random.seed!(1234321)
    refit!(simulate!(fm))
    @test deviance(fm) ≈ 339.0218639362958 atol=0.001
    simulate!(fm, θ = fm.θ)
    @test_throws DimensionMismatch refit!(fm, zeros(29))
    bsamp = parametricbootstrap(MersenneTwister(1234321), 10, fm)
    @test length(bsamp.objective) == 10
    @test keys(bsamp.bstr) == (:objective, :σ, :β₁, :θ)
end

@testset "Rank deficient" begin
    Random.seed!(0)
    x = rand(100)
    data = columntable((x = x, x2 = 1.5 .* x, y = rand(100), z = repeat('A':'T', 5)))
    model = fit!(LinearMixedModel(@formula(y ~ x + x2 + (1|z)), data))
    @test length(fixef(model)) == 2
    @test rank(model) == 2
    @test length(coef(model)) == 3
end
