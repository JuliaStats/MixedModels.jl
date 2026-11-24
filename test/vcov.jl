using DataFrames
using LinearAlgebra
using MixedModels
using MixedModels: dataset, fullrankx
using SparseArrays
using StableRNGs
using StatsAPI: crossmodelmatrix
using Test
using Vcov

# Dense reference quantities in the pivoted, full-rank (and weighted) space:
# relative marginal covariance V = I + ZΛΛ'Z', X, and r = y - Xβ̂.
function dense_reference(m::LinearMixedModel)
    sqrtw = isempty(m.sqrtwts) ? ones(nobs(m)) : m.sqrtwts
    ZΛ = reduce(hcat,
        [Matrix(sparse(t)) * kron(I(MixedModels.nlevs(t)), Matrix(t.λ))
         for t in m.reterms])
    ZΛ .*= sqrtw
    V = I + ZΛ * ZΛ'
    X = sqrtw .* Matrix(fullrankx(m))
    β = fixef(m) # pivoted, truncated to rank(m)
    r = sqrtw .* (response(m) - Matrix(fullrankx(m)) * β)
    return (; V, X, r)
end

sleepstudy = DataFrame(dataset(:sleepstudy))
sleepstudy.days2 = 2 .* sleepstudy.days
sleepstudy.wts = rand(StableRNG(42), 0.5:0.5:2.0, nrow(sleepstudy))

fm = fit(MixedModel, @formula(reaction ~ 1 + days + (1 + days | subj)), sleepstudy;
    progress=false)
fm_rankdef = fit(MixedModel, @formula(reaction ~ 1 + days + days2 + (1 | subj)),
    sleepstudy; progress=false)
fm_wts = fit(MixedModel, @formula(reaction ~ 1 + days + (1 + days | subj)), sleepstudy;
    wts=sleepstudy.wts, progress=false)
fm_crossed = fit(MixedModel,
    @formula(rt_trunc ~ 1 + spkr * prec + (1 | subj) + (1 | item)), dataset(:kb07);
    progress=false)
gm = fit(MixedModel, @formula(r2 ~ 1 + anger + gender + (1 | subj)), dataset(:verbagg),
    Bernoulli(); progress=false)

@testset "VcovData" begin
    @testset "$(name)" for (name, m) in
                           ("simple" => fm, "rank deficient" => fm_rankdef,
        "weighted" => fm_wts, "crossed" => fm_crossed)
        (; V, X, r) = dense_reference(m)
        vd = Vcov.VcovData(m)
        @test modelmatrix(vd) ≈ X
        @test crossmodelmatrix(vd) ≈ X' * (V \ X)
        @test Vcov.invcrossmodelmatrix(vd) ≈ inv(X' * (V \ X))
        # V⁻¹(y - Xβ̂) are the conditional residuals
        @test residuals(vd) ≈ V \ r atol = 1e-8
        @test dof_residual(vd) == nobs(m) - rank(m)
    end

    @test_throws MethodError Vcov.VcovData(gm)
end

@testset "simple" begin
    @testset "$(name)" for (name, m) in
                           ("simple" => fm, "rank deficient" => fm_rankdef,
        "weighted" => fm_wts, "crossed" => fm_crossed)
        @test isequal(vcov(m, Vcov.simple()), vcov(m))
    end
    @test_throws MethodError vcov(gm, Vcov.simple())
end

# compare an estimator's result to a reference computed in the pivoted, full-rank space
function test_against_reference(m, est, ref)
    vc = vcov(m, est)
    @test size(vc) == size(vcov(m))
    @test isnan.(vc) == isnan.(vcov(m))
    piv = m.feterm.piv
    r = rank(m)
    @test vc[piv, piv][1:r, 1:r] ≈ ref
    return nothing
end

@testset "robust (HC1)" begin
    @testset "$(name)" for (name, m) in
                           ("simple" => fm, "rank deficient" => fm_rankdef,
        "weighted" => fm_wts, "crossed" => fm_crossed)
        (; V, X, r) = dense_reference(m)
        B = inv(X' * (V \ X))
        scores = X .* (V \ r)
        n, p = size(X)
        ref = B * (n / (n - p) * scores' * scores) * B
        test_against_reference(m, Vcov.robust(), ref)
        test_against_reference(m, Vcov.hc(1), ref)
    end
    @test_throws ArgumentError vcov(fm, Vcov.hc(2))
    @test_throws ArgumentError vcov(fm, Vcov.hc(3))
    @test_throws MethodError vcov(gm, Vcov.robust())
end

# sum of outer products of the within-cluster score sums
function cluster_meat(scores, g)
    S = zeros(size(scores, 2), size(scores, 2))
    for k in unique(g)
        s = vec(sum(scores[g .== k, :]; dims=1))
        S += s * s'
    end
    return S
end

@testset "cluster (CR1)" begin
    @testset "one-way: $(name)" for (name, m) in
                                    ("simple" => fm, "rank deficient" => fm_rankdef,
        "weighted" => fm_wts, "crossed" => fm_crossed)
        (; V, X, r) = dense_reference(m)
        B = inv(X' * (V \ X))
        scores = X .* (V \ r)
        n, p = size(X)
        g = m.reterms[1].refs
        G = length(unique(g))
        ref = B * ((n - 1) / (n - p) * G / (G - 1) * cluster_meat(scores, g)) * B
        test_against_reference(m, Vcov.cluster(MixedModels.fnames(m)[1]), ref)
    end

    @testset "two-way" begin
        # With crossed grouping factors the two-way meat is generally not positive
        # semidefinite and Vcov applies the Cameron-Gelbach-Miller eigenvalue
        # adjustment. So we check the meat (before adjustment) against the
        # inclusion-exclusion formula, then check that vcov returns the adjusted matrix.
        m = fm_crossed
        (; V, X, r) = dense_reference(m)
        scores = X .* (V \ r)
        n, p = size(X)
        gs, gi = (t.refs for t in m.reterms)
        gsi = Int.(gs) .* (maximum(gi) + 1) .+ gi # intersection of subj and item
        G = min(length(unique(gs)), length(unique(gi)))
        meat =
            cluster_meat(scores, gs) + cluster_meat(scores, gi) -
            cluster_meat(scores, gsi)
        ref = (n - 1) / (n - p) * G / (G - 1) * meat
        ext = Base.get_extension(MixedModels, :MixedModelsVcovExt)
        est = ext._materialize(m, Vcov.cluster(:subj, :item))
        @test Vcov.S_hat(Vcov.VcovData(m), est) ≈ ref

        vc = @test_logs (:warn, r"not positive-semidefinite") vcov(m,
            Vcov.cluster(:subj, :item))
        @test size(vc) == size(vcov(m))
        # the adjustment clamps negative eigenvalues to zero: PSD, possibly singular
        @test minimum(eigvals(Symmetric(vc))) >= -sqrt(eps()) * opnorm(vc)
    end

    @testset "materialized clusters" begin
        # same as clustering by name
        g = Vcov.GroupedArray(sleepstudy.subj)
        @test vcov(fm, Vcov.cluster((:subj,), (g,))) ≈ vcov(fm, Vcov.cluster(:subj))
        # a variable that is not a grouping factor
        g = Vcov.GroupedArray(sleepstudy.days)
        @test all(isfinite, vcov(fm, Vcov.cluster((:days,), (g,))))
        @test_throws DimensionMismatch vcov(fm,
            Vcov.cluster((:subj,), (Vcov.GroupedArray(sleepstudy.subj[1:10]),)))
    end

    @test_throws ArgumentError vcov(fm, Vcov.cluster(:days))
    @test_throws MethodError vcov(gm, Vcov.cluster(:subj))
end

@testset "stderror" begin
    @testset "$(name)" for (name, m) in
                           ("simple" => fm, "rank deficient" => fm_rankdef,
        "weighted" => fm_wts, "crossed" => fm_crossed)
        @test stderror(m, Vcov.simple()) ≈ stderror(m) nans = true
        @testset "$(est)" for est in (Vcov.robust(), Vcov.cluster(:subj))
            se = stderror(m, est)
            @test length(se) == length(coef(m))
            @test isequal(isnan.(se), isnan.(stderror(m)))
            @test se ≈ sqrt.(diag(vcov(m, est))) nans = true
        end
    end
    @test_throws MethodError stderror(gm, Vcov.robust())
end
