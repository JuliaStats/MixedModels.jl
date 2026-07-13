using DataFrames
using LinearAlgebra
using MixedModels
using Random
using SparseArrays
using StableRNGs
using Statistics
using Tables
using Test

using MixedModels: ismultimember, isnested

# profiled ML deviance computed from the dense marginal covariance matrix
function dense_objective(m::LinearMixedModel)
    X = m.X
    y = m.y
    n = length(y)
    V = zeros(n, n)
    for (rt, λ) in zip(m.reterms, m.λ)
        Z = Matrix(rt)
        Λ = kron(I(MixedModels.nlevs(rt)), Matrix(λ))
        V .+= Z * Λ * Λ' * Z'
    end
    cf = cholesky!(Symmetric(V + I))
    β = (X' * (cf \ X)) \ (X' * (cf \ y))
    r = y - X * β
    σ² = (r' * (cf \ r)) / n
    return n * log(2π * σ²) + logdet(cf) + n
end

function mmdata(rng; nobs=500, nm=15, slopes=false)
    W = Float64.(rand(rng, nobs, nm) .< 0.25)
    for i in 1:nobs  # ensure every observation has at least one membership
        if all(iszero, view(W, i, :))
            W[i, rand(rng, 1:nm)] = 1.0
        end
    end
    x = randn(rng, nobs)
    y = 1 .+ 2 .* x + W * (1.5 .* randn(rng, nm)) + randn(rng, nobs)
    if slopes
        y .+= (W * (0.5 .* randn(rng, nm))) .* x
    end
    return (; x, y), W
end

@testset "membershipmatrix helpers" begin
    mm = membershipmatrix(["a,b", "b , c", "a", "a,a"])
    @test mm.levels == ["a", "b", "c"]
    @test Matrix(mm.weights) == [1 0 1 2; 1 1 0 0; 0 1 0 0]
    @test size(mm) == (3, 4)
    @test size(mm, 2) == 4
    @test levels(mm) == ["a", "b", "c"]
    @test sprint(show, MIME("text/plain"), mm) isa String

    mmnorm = membershipmatrix(["a,b", "a,b,c,d"]; normalize=true)
    @test all(sum(Matrix(mmnorm.weights); dims=1) .≈ 1)

    mmsemi = membershipmatrix(["a;b", "c"]; delim=";")
    @test mmsemi.levels == ["a", "b", "c"]

    mmcols = membershipmatrix(["x", "y", missing], ["y", missing, missing])
    @test mmcols.levels == ["x", "y"]
    @test Matrix(mmcols.weights) == [1 0 0; 1 1 0]
    @test_throws DimensionMismatch membershipmatrix(["x"], ["y", "z"])

    mmint = interactionweights(mmcols, membershipmatrix(["u", "u", "v"]))
    @test mmint.levels == ["x & u", "x & v", "y & u", "y & v"]
    @test size(mmint.weights) == (4, 3)
    @test_throws DimensionMismatch interactionweights(mmcols, membershipmatrix(["u"]))

    @test_throws DimensionMismatch MembershipMatrix(ones(2, 3); levels=["a"])
    @test_throws ArgumentError MembershipMatrix(ones(2, 3); levels=["a", "a"])
end

@testset "equivalence with single membership" begin
    rng = StableRNG(42)
    nobs = 400
    g = rand(rng, string.('a':'j'), nobs)
    x = randn(rng, nobs)
    b = Dict(l => 1.5 * randn(rng) for l in unique(g))
    s = Dict(l => 0.5 * randn(rng) for l in unique(g))
    y = 1 .+ 2 .* x .+ [b[gi] + s[gi] * xi for (gi, xi) in zip(g, x)] .+
        randn(rng, nobs)
    df = (; x, y, g)
    mmw = membershipmatrix(df.g)
    wts = ones(nobs) .+ rand(rng, nobs)

    @testset "$(form)" for (form, mmform) in (
        (@formula(y ~ 1 + x + (1 | g)), @formula(y ~ 1 + x + (1 | gmm))),
        (@formula(y ~ 1 + x + (1 + x | g)), @formula(y ~ 1 + x + (1 + x | gmm))),
        (
            @formula(y ~ 1 + x + zerocorr(1 + x | g)),
            @formula(y ~ 1 + x + zerocorr(1 + x | gmm)),
        ),
    )
        mref = fit(MixedModel, form, df; progress=false)
        mequiv = fit(
            MixedModel, mmform, df; memberships=Dict(:gmm => mmw), progress=false
        )
        @test objective(mref) ≈ objective(mequiv) rtol = 1e-8
        @test mref.θ ≈ mequiv.θ atol = 1e-4
        @test coef(mref) ≈ coef(mequiv) rtol = 1e-6
        @test stderror(mref) ≈ stderror(mequiv) rtol = 1e-5
        @test only(ranef(mref)) ≈ only(ranef(mequiv)) atol = 1e-4
        @test fitted(mref) ≈ fitted(mequiv) atol = 1e-4
    end

    @testset "weighted LMM" begin
        mref = fit(
            MixedModel, @formula(y ~ 1 + x + (1 | g)), df; weights=wts, progress=false
        )
        mequiv = fit(MixedModel, @formula(y ~ 1 + x + (1 | gmm)), df;
            weights=wts, memberships=Dict(:gmm => mmw), progress=false)
        @test objective(mref) ≈ objective(mequiv) rtol = 1e-8
        @test mref.θ ≈ mequiv.θ atol = 1e-4
        @test coef(mref) ≈ coef(mequiv) rtol = 1e-6
    end

    @testset "Bernoulli GLMM" begin
        ybin = rand(rng, nobs) .< 1 ./ (1 .+ exp.(-(x .+ [b[gi] for gi in g])))
        dfb = (; x, y=ybin, g)
        mref = fit(
            MixedModel, @formula(y ~ 1 + x + (1 | g)), dfb, Bernoulli(); progress=false
        )
        mequiv = fit(MixedModel, @formula(y ~ 1 + x + (1 | gmm)), dfb, Bernoulli();
            memberships=Dict(:gmm => mmw), progress=false)
        @test deviance(mref) ≈ deviance(mequiv) rtol = 1e-6
        @test mref.θ ≈ mequiv.θ atol = 1e-4
        @test coef(mref) ≈ coef(mequiv) rtol = 1e-4
    end
end

@testset "genuine multimembership LMM" begin
    rng = StableRNG(1)
    df, W = mmdata(rng)
    mms = Dict(:members => MembershipMatrix(W'))
    m = fit(MixedModel, @formula(y ~ 1 + x + (1 | members)), df;
        memberships=mms, progress=false)
    @test ismultimember(only(m.reterms))
    @test MixedModels.nlevs(only(m.reterms)) == 15
    @test m.θ == MixedModels.getθ(only(m.reterms))
    @test objective(m) ≈ dense_objective(m) rtol = 1e-6

    # the fitted θ is a local minimum of the dense objective
    mpert = deepcopy(m)
    for f in (0.9, 0.95, 1.05, 1.1)
        MixedModels.updateL!(MixedModels.setθ!(mpert, f .* m.θ))
        @test dense_objective(mpert) > objective(m)
    end

    # fitted values decompose as Xβ + Zb
    Z = Matrix(only(m.reterms))
    @test fitted(m) ≈ m.X * coef(m) + Z * vec(only(ranef(m))) rtol = 1e-8

    # basic show/accessor smoke tests
    @test sprint(show, MIME("text/plain"), m) isa String
    @test sprint(show, MIME("text/plain"), VarCorr(m)) isa String
    @test sprint(show, MIME("text/plain"), BlockDescription(m)) isa String
    @test only(propertynames(raneftables(m))) == :members
    @test length(condVar(m)) == 1
    @test size(only(condVar(m))) == (1, 1, 15)

    # raw (non-MembershipMatrix) weight matrices are accepted
    m2 = fit(MixedModel, @formula(y ~ 1 + x + (1 | members)), df;
        memberships=Dict(:members => W'), progress=false)
    @test objective(m2) ≈ objective(m)

    # leverage and cooksdistance are not supported
    @test_throws ArgumentError leverage(m)
    @test_throws ArgumentError cooksdistance(m)
end

@testset "random slopes" begin
    rng = StableRNG(2)
    df, W = mmdata(rng; slopes=true)
    m = fit(MixedModel, @formula(y ~ 1 + x + (1 + x | members)), df;
        memberships=Dict(:members => W'), progress=false)
    @test objective(m) ≈ dense_objective(m) rtol = 1e-6
    Z = Matrix(only(m.reterms))
    @test fitted(m) ≈ m.X * coef(m) + Z * vec(only(ranef(m))) rtol = 1e-8
    vc = VarCorr(m)
    @test sprint(show, MIME("text/plain"), vc) isa String
end

@testset "crossed with single-membership term" begin
    rng = StableRNG(3)
    df, W = mmdata(rng)
    subj = rand(rng, string.(1:30), 500)
    y = df.y .+ [randn(StableRNG(hash(s) % 100000)) for s in subj]
    df = (; df.x, y, subj)
    m = fit(MixedModel, @formula(y ~ 1 + x + (1 | subj) + (1 | members)), df;
        memberships=Dict(:members => W'), progress=false)
    # multimembership term sorts last; single-membership block stays Diagonal
    @test !ismultimember(first(m.reterms))
    @test ismultimember(last(m.reterms))
    @test first(m.L) isa Diagonal
    @test objective(m) ≈ dense_objective(m) rtol = 1e-6
    @test !isnested(first(m.reterms), last(m.reterms))
    @test !isnested(last(m.reterms), first(m.reterms))
    @test !isnested(last(m.reterms), last(m.reterms))
end

@testset "genuine multimembership GLMM" begin
    rng = StableRNG(4)
    nobs = 800
    nm = 15
    W = Float64.(rand(rng, nobs, nm) .< 0.2)
    for i in 1:nobs
        if all(iszero, view(W, i, :))
            W[i, rand(rng, 1:nm)] = 1.0
        end
    end
    x = randn(rng, nobs)
    bmm = randn(rng, nm)
    y = rand(rng, nobs) .< 1 ./ (1 .+ exp.(-(0.5 .+ x .+ W * bmm)))
    df = (; x, y)
    mms = Dict(:members => W')
    m = fit(MixedModel, @formula(y ~ 1 + x + (1 | members)), df, Bernoulli();
        memberships=mms, progress=false)
    @test isfinite(deviance(m))
    @test cor(vec(only(ranef(m))), bmm) > 0.7

    mfast = fit(MixedModel, @formula(y ~ 1 + x + (1 | members)), df, Bernoulli();
        memberships=mms, fast=true, progress=false)
    @test isfinite(deviance(mfast))

    @test_throws ArgumentError fit(
        MixedModel, @formula(y ~ 1 + x + (1 | members)), df, Bernoulli();
        memberships=mms, nAGQ=3, progress=false,
    )
    @test_throws ArgumentError deviance(m, 3)

    bs = parametricbootstrap(StableRNG(5), 20, m; progress=false)
    @test all(isfinite, bs.objective)
end

@testset "predict and simulate" begin
    rng = StableRNG(6)
    df, W = mmdata(rng)
    mms = Dict(:members => MembershipMatrix(W'))
    m = fit(MixedModel, @formula(y ~ 1 + x + (1 | members)), df;
        memberships=mms, progress=false)

    @test predict(m, df; memberships=mms, new_re_levels=:error) ≈ fitted(m) rtol = 1e-8
    sub = 1:100
    dfsub = (; x=df.x[sub], y=df.y[sub])
    mmsub = Dict(:members => MembershipMatrix(W[sub, :]'))
    @test predict(m, dfsub; memberships=mmsub, new_re_levels=:error) ≈
        fitted(m)[sub] rtol = 1e-8
    @test_throws ArgumentError predict(m, dfsub)

    # a membership matrix with an unseen level
    Wnew = vcat(W[sub, :]', zeros(1, length(sub)))
    Wnew[end, 1] = 1.0
    mmnew = Dict(:members => MembershipMatrix(Wnew; levels=[string.(1:15); "new"]))
    @test_throws ArgumentError predict(m, dfsub; memberships=mmnew, new_re_levels=:error)
    pmiss = predict(m, dfsub; memberships=mmnew, new_re_levels=:missing)
    @test ismissing(pmiss[1])
    @test pmiss[2:end] ≈ fitted(m)[2:100] rtol = 1e-8
    ppop = predict(m, dfsub; memberships=mmnew, new_re_levels=:population)
    @test ppop[2:end] ≈ fitted(m)[2:100] rtol = 1e-8
    @test !ismissing(ppop[1])

    ysim = simulate(StableRNG(1), m)
    @test length(ysim) == length(df.y)
    @test_throws ArgumentError simulate!(StableRNG(1), similar(df.y[sub]), m, dfsub)
    ysub = simulate!(
        StableRNG(1), similar(df.y[sub]), m, dfsub; memberships=mmsub
    )
    @test length(ysub) == 100

    bs = parametricbootstrap(StableRNG(7), 50, m; progress=false)
    @test all(isfinite, bs.objective)
    ci = shortestcovint(bs)
    @test length(ci) == 4

    m2 = refit!(deepcopy(m), df.y; progress=false)
    @test objective(m2) ≈ objective(m) rtol = 1e-8
end

@testset "missing data alignment" begin
    rng = StableRNG(8)
    df, W = mmdata(rng; nobs=300)
    x = convert(Vector{Union{Float64,Missing}}, df.x)
    x[[10, 20, 30]] .= missing
    dfmiss = (; x, df.y)
    keep = map(!ismissing, x)
    mms = Dict(:members => MembershipMatrix(W'))
    m = fit(MixedModel, @formula(y ~ 1 + x + (1 | members)), dfmiss;
        memberships=mms, progress=false)
    @test nobs(m) == 297
    mref = fit(MixedModel, @formula(y ~ 1 + x + (1 | members)),
        (; x=df.x[keep], y=df.y[keep]);
        memberships=Dict(:members => MembershipMatrix(W[keep, :]')), progress=false)
    @test objective(m) ≈ objective(mref) rtol = 1e-8
end

@testset "error paths" begin
    rng = StableRNG(9)
    df, W = mmdata(rng; nobs=200)
    dfg = (; df.x, df.y, g=rand(rng, ["u", "v"], 200))
    mms = Dict(:members => MembershipMatrix(W'))

    # membership key is an existing column
    @test_throws ArgumentError fit(
        MixedModel, @formula(y ~ 1 + x + (1 | g)), dfg;
        memberships=Dict(:g => W'), progress=false,
    )
    # membership key matches no grouping variable
    @test_throws ArgumentError fit(
        MixedModel, @formula(y ~ 1 + x + (1 | g)), dfg;
        memberships=mms, progress=false,
    )
    # observation-count mismatch
    @test_throws DimensionMismatch fit(
        MixedModel, @formula(y ~ 1 + x + (1 | members)), dfg;
        memberships=Dict(:members => W[1:100, :]'), progress=false,
    )
    # two terms sharing the multimembership grouping name cannot be amalgamated
    @test_throws ArgumentError fit(
        MixedModel, @formula(y ~ 1 + x + (1 | members) + (0 + x | members)), dfg;
        memberships=mms, progress=false,
    )
end
