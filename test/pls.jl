using Compat, StatsBase, DataFrames, RData, MixedModels
using Compat.Test
using CategoricalArrays

if !isdefined(:dat) || !isa(dat, Dict{Symbol, Any})
    dat = convert(Dict{Symbol,Any}, load(joinpath(dirname(@__FILE__), "dat.rda")))
end

@testset "Dyestuff" begin
    fm1 = lmm(@formula(Y ~ 1 + (1|G)), dat[:Dyestuff])

    @test nblocks(fm1.A) == (3, 3)
    @test size(fm1.trms) == (3, )
    @test nblocks(fm1.L.data) == (3, 3)
    @test lowerbd(fm1) == zeros(1)
    @test getθ(fm1) == ones(1)
    @test getΛ(fm1) == [1.0]

    @test objective(updateL!(setθ!(fm1, [0.713]))) ≈ 327.34216280955366
    MixedModels.describeblocks(IOBuffer(), fm1)

    fit!(fm1);
    @test isapprox(objective(fm1), 327.3270598811428, atol=0.001)
    @test isapprox(getθ(fm1), [0.752580], atol=1.e-5)
    @test isapprox(deviance(fm1), 327.32705988, atol=0.001)
    @test isapprox(aic(fm1), 333.3270598811394, atol=0.001)
    @test isapprox(bic(fm1), 337.5306520261259, atol=0.001)
    @test fixef(fm1) ≈ [1527.5]
    @test StatsBase.dof(fm1) == 3
    @test StatsBase.nobs(fm1) == 30
    @test MixedModels.fixef!(zeros(1),fm1) ≈ [1527.5]
    @test coef(fm1) ≈ [1527.5]
    @test cond(fm1) == ones(1)
    cm = coeftable(fm1)
    @test length(cm.rownms) == 1
    @test length(cm.colnms) == 4
    @test MixedModels.fnames(fm1) == [:G]
    @test model_response(fm1) == Vector(dat[:Dyestuff][:Y])
    rfu = ranef(fm1, uscale = true)
    rfb = ranef(fm1)
    cv = condVar(fm1)
    @test abs(sum(rfu[1])) < 1.e-5
    @test length(cv) == 1
    @test size(cv[1]) == (1, 1, 6)
    show(IOBuffer(), fm1.optsum)

    @test isapprox(logdet(fm1), 8.06014522999825, atol=0.001)
    @test isapprox(varest(fm1), 2451.2501089607676, atol=0.001)
    @test isapprox(pwrss(fm1), 73537.49947885796, atol=0.001)
    @test isapprox(stderr(fm1), [17.69455188898009], atol=0.0001)

    vc = VarCorr(fm1)
    show(IOBuffer(), vc)
    @test vc.s == sdest(fm1)
end

@testset "Dyestuff2" begin
    fm = lmm(@formula(Y ~ 1 + (1 | G)), dat[:Dyestuff2])
    @test lowerbd(fm) == zeros(1)
    fit!(fm, true)
    show(IOBuffer(), fm)
    @test getθ(fm)[1] < 1.0e-9
    @test objective(fm) ≈ 162.87303665382575
    @test abs(std(fm)[1][1]) < 1.0e-9
    @test std(fm)[2] ≈ [3.653231351374652]
    @test stderr(fm) ≈ [0.6669857396443261]
    @test coef(fm) ≈ [5.6656]
    @test logdet(fm) ≈ 0.0
    refit!(fm, dat[:Dyestuff][:Y])
    @test isapprox(objective(fm), 327.3270598811428, atol=0.001)
end

@testset "penicillin" begin
    fm = lmm(@formula(Y ~ 1 + (1 | G) + (1 | H)), dat[:Penicillin]);
    @test size(fm) == (144, 1, 30, 2)
    @test getθ(fm) == ones(2)
    @test lowerbd(fm) == zeros(2)

    fit!(fm)

    @test isapprox(objective(fm), 332.18834867227616, atol=0.001)
    @test isapprox(coef(fm), [22.97222222222222], atol=0.001)
    @test isapprox(fixef(fm), [22.97222222222222], atol=0.001)
    @test coef(fm)[1] ≈ mean(dat[:Penicillin][:Y])
    @test isapprox(stderr(fm), [0.7445960346851368], atol=0.0001)
    @test isapprox(getθ(fm), [1.5375772376554968, 3.219751321180035], atol=0.001)
    @test isapprox(std(fm)[1], [0.8455645948223015], atol=0.0001)
    @test isapprox(std(fm)[2], [1.770647779277388], atol=0.0001)
    @test isapprox(varest(fm), 0.3024263987592062, atol=0.0001)
    @test isapprox(logdet(fm), 95.74614821367786, atol=0.001)
    rfu = ranef(fm, uscale=true)
    rfb = ranef(fm)
    @test length(rfb) == 2
end

@testset "pastes" begin
    fm = lmm(@formula(Y ~ (1 | G) + (1 | H)), dat[:Pastes])
    @test size(fm) == (60, 1, 40, 2)
    @test getθ(fm) == ones(2)
    @test lowerbd(fm) == zeros(2)

    fit!(fm);

    @test isapprox(objective(fm), 247.99446586289676, atol=0.001)
    @test isapprox(coef(fm), [60.05333333333329], atol=0.001)
    @test isapprox(fixef(fm), [60.05333333333329], atol=0.001)
    @test isapprox(stderr(fm), [0.6421359883527029] , atol=0.0001)
    @test isapprox(getθ(fm), [3.5268858714382905, 1.3299230213750168], atol=0.001)
    @test isapprox(std(fm)[1], [2.904069002535747], atol=0.001)
    @test isapprox(std(fm)[2], [1.095070371687089], atol=0.0001)
    @test isapprox(std(fm)[3], [0.8234088395243269], atol=0.0001)
    @test isapprox(varest(fm), 0.6780020742644107, atol=0.0001)
    @test isapprox(logdet(fm), 101.0381339953986, atol=0.001)
end

@testset "InstEval" begin
    df = dat[:InstEval]
    fm1 = lmm(@formula(Y ~ 1 + A + (1 | G) + (1 | H) + (1 | I)), df)
    @test size(fm1) == (73421, 2, 4114, 3)
    @test getθ(fm1) == ones(3)
    @test lowerbd(fm1) == zeros(3)

    fit!(fm1);

    @test isapprox(objective(fm1), 237721.7687745563, atol=0.001)
    ftd1 = fitted(fm1);
    @test size(ftd1) == (73421, )
    @test ftd1 == predict(fm1)
    
    # fit with a subset of the data
    # set the categoricals first, so the levels will be copied
    df[:G] = categorical(df[:G])
    df[:H] = categorical(df[:H])
    df[:I] = categorical(df[:I])
    subset = df[1:1000,:]
    println(size(df))
    println(size(subset))
    ftd2 = predict(fm1,subset)
    @test ftd2 == ftd1[1:1000]
    
    @test isapprox(ftd1[1], 3.17876, atol=0.0001)
    resid1 = residuals(fm1);
    @test size(resid1) == (73421, )
    @test isapprox(resid1[1], 1.82124, atol=0.00001)

    fm2 = fit!(lmm(@formula(Y ~ 1 + A*I + (1 | G) + (1 | H)), dat[:InstEval]))
    @test isapprox(objective(fm2), 237585.5534151694, atol=0.001)
    @test size(fm2) == (73421, 28, 4100, 2)
end

@testset "sleep" begin
    fm = lmm(@formula(Y ~ 1 + U + (1 + U | G)), dat[:sleepstudy]);
    @test lowerbd(fm) == [0.0, -Inf, 0.0]
    A11 = fm.A[Block(1,1)]
    @test isa(A11, UniformBlockDiagonal{Float64})
    @test isa(fm.L.data[Block(1, 1)], UniformBlockDiagonal{Float64})
    @test size(A11) == (36, 36)
    @test A11.facevec[1] == [10. 45.; 45. 285.]
    @test length(A11.facevec) == 18
    updateL!(fm);
    b11 = LowerTriangular(fm.L.data[Block(1, 1)].facevec[1])
    @test b11 * b11' == fm.A[Block(1, 1)].facevec[1] + I
    @test countnz(full(fm.L.data[Block(1, 1)])) == 18 * 4


    fit!(fm)

    @test objective(fm) ≈ 1751.9393444647046
    @test isapprox(getθ(fm), [0.929221307, 0.01816838, 0.22264487096], atol=1.e-6)
    @test isapprox(pwrss(fm), 117889.46144025437)
    @test isapprox(logdet(fm), 73.90322021999222, atol=0.001)
    @test isapprox(stderr(fm), [6.632257721914501, 1.5022354739749826], atol=0.0001)
    @test coef(fm) ≈ [251.40510484848477,10.4672859595959]
    @test fixef(fm) ≈ [10.4672859595959, 251.40510484848477]
    @test isapprox(stderr(fm), [6.632246393963571, 1.502190605041084], atol=0.01)
    @test isapprox(std(fm)[1], [23.780468100188497, 5.716827903196682], atol=0.01)
    @test isapprox(logdet(fm), 73.90337187545992, atol=0.001)
    @test diag(cor(fm)[1]) ≈ ones(2)
    @test isapprox(cond(fm), [4.175251], atol=0.0001)
    @test loglikelihood(fm) ≈ -875.9696722323523
    show(IOBuffer(), fm)

    u3 = ranef(fm, uscale=true)
    @test length(u3) == 1
    @test size(u3[1]) == (2, 18)
    @test isapprox(u3[1][1, 1], 3.030300122575336, atol=0.001)
    u3n = ranef(fm, uscale=true, named=true)

    b3 = ranef(fm)
    @test length(b3) == 1
    @test size(b3[1]) == (2, 18)
    @test isapprox(b3[1][1, 1], 2.815819441982976, atol=0.001)

    simulate!(fm)  # to test one of the unscaledre methods

    fmnc = lmm(@formula(Y ~ 1 + U + (1|G) + (0+U|G)), dat[:sleepstudy])
    @test size(fmnc) == (180,2,36,1)
    @test getθ(fmnc) == ones(2)
    @test lowerbd(fmnc) == zeros(2)

    fit!(fmnc)

    @test isapprox(deviance(fmnc), 1752.0032551398835, atol=0.001)
    @test isapprox(objective(fmnc), 1752.0032551398835, atol=0.001)
    @test coef(fmnc) ≈ [251.40510484848585, 10.467285959595715]
    @test fixef(fmnc) ≈ [10.467285959595715, 251.40510484848477]
    @test isapprox(stderr(fmnc), [6.707710260366577, 1.5193083237479683], atol=0.001)
    @test isapprox(getθ(fmnc), [0.9458106880922268, 0.22692826607677266], atol=0.0001)
    @test std(fmnc)[1] ≈ [24.171449463289047, 5.799379721123582]
    @test std(fmnc)[2] ≈ [25.556130034081047]
    @test isapprox(logdet(fmnc), 74.46952585564611, atol=0.001)
    cor(fmnc)

    MixedModels.lrt(fm, fmnc)

    fmrs = fit!(lmm(@formula(Y ~ 1 + U + (0 + U|G)), dat[:sleepstudy]))
    @test isapprox(objective(fmrs), 1774.080315280528, rtol=0.00001)
    @test isapprox(getθ(fmrs), [0.24353985679033105], rtol=0.00001)   
end

@testset "d3" begin
    fm = updateL!(lmm(@formula(Y ~ 1 + U + (1+U|G) + (1+U|H) + (1+U|I)), dat[:d3]));
    @test isapprox(pwrss(fm), 5.1261847180180885e6, rtol = 1e-6)
    @test isapprox(logdet(fm), 52718.0137366602, rtol = 1e-6)
    @test isapprox(objective(fm), 901641.2930413672, rtol = 1e-6)
    fit!(fm)
    @test isapprox(objective(fm), 884957.5540213, rtol = 1e-6)
    @test isapprox(coef(fm), [0.4991229873, 0.31130780953], atol = 1.e-4)
end

@testset "simulate!" begin
    @test MixedModels.stddevcor(cholfact!(eye(3))) == (ones(3), eye(3))
    fm = fit!(lmm(@formula(Y ~ 1 + (1 | G)), dat[:Dyestuff]))
    refit!(simulate!(MersenneTwister(1234321), fm))
    @test isapprox(deviance(fm), 339.0218639362958, atol=0.001)
    refit!(fm, dat[:Dyestuff][:Y])
    srand(1234321)
    refit!(simulate!(fm))
    @test isapprox(deviance(fm), 339.0218639362958, atol=0.001)
    simulate!(fm, θ = getθ(fm))
    @test_throws DimensionMismatch refit!(fm, zeros(29))
    srand(1234321)
    dfr = bootstrap(10, fm)
    @test size(dfr) == (10, 5)
    @test names(dfr) == Symbol[:obj, :σ, :β₁, :θ₁, :σ₁]
end
