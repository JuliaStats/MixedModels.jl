using DataFrames
using LinearAlgebra
using MixedModels
using MixedModels: dataset, fullrankx, pivot
using SparseArrays
using StableRNGs
using StatsAPI: crossmodelmatrix
using StatsBase: SimpleCovariance
using Test
using Vcov

# Dense reference quantities in the pivoted, full-rank (and weighted) space:
# relative marginal covariance V = I + ZΛΛ'Z', X, and r = y - Xβ̂.
function dense_reference(m::LinearMixedModel)
    sqrtw = sqrt.(weights(m))
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

# LinearMixedModels covering the cases the estimators need to handle,
# keyed by the name used for their testsets
fm = lmm(
    @formula(reaction ~ 1 + days + (1 + days | subj)),
    sleepstudy;
    progress=false,
)
fm_rankdef = lmm(
    @formula(reaction ~ 1 + days + days2 + (1 | subj)),
    sleepstudy;
    progress=false,
)
fm_wts = lmm(
    @formula(reaction ~ 1 + days + (1 + days | subj)),
    sleepstudy;
    wts=sleepstudy.wts,
    progress=false,
)
fm_crossed = lmm(
    @formula(rt_trunc ~ 1 + spkr * prec + (1 | subj) + (1 | item)),
    dataset(:kb07);
    progress=false,
)
lmms = Dict{String,MixedModel}(
    "simple" => fm,
    "rank deficient" => fm_rankdef,
    "weighted" => fm_wts,
    "crossed" => fm_crossed,
)
gm = fit(MixedModel, @formula(r2 ~ 1 + anger + gender + (1 | subj)), dataset(:verbagg),
    Bernoulli(); progress=false)

@testset "VcovData" begin
    @testset "$(name)" for (name, m) in lmms
        (; V, X, r) = dense_reference(m)
        vd = Vcov.VcovData(m)
        @test modelmatrix(vd) ≈ X
        @test crossmodelmatrix(vd) ≈ X' * (V \ X)
        @test Vcov.invcrossmodelmatrix(vd) ≈ inv(X' * (V \ X))
        # V⁻¹(y - Xβ̂) are the conditional residuals
        @test residuals(vd) ≈ V \ r atol = 1e-8
        @test dof_residual(vd) == nobs(m) - rank(m)
    end

end

@testset "simple" begin
    @testset "$(name)" for (name, m) in lmms
        @test isequal(vcov(m, Vcov.simple()), vcov(m))
    end
end

# compare an estimator's result to a reference computed in the pivoted, full-rank space
function test_against_reference(m, est, ref)
    vc = vcov(m, est)
    @test size(vc) == size(vcov(m))
    @test isnan.(vc) == isnan.(vcov(m))
    piv = pivot(m)
    r = rank(m)
    @test vc[piv, piv][1:r, 1:r] ≈ ref
    return nothing
end

@testset "robust (HC1)" begin
    @testset "$(name)" for (name, m) in lmms
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
    @testset "one-way: $(name)" for (name, m) in lmms
        (; V, X, r) = dense_reference(m)
        B = inv(X' * (V \ X))
        scores = X .* (V \ r)
        n, p = size(X)
        g = refarray(m.reterms[1])
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
        gs, gi = (refarray(t) for t in m.reterms)
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
end

@testset "stderror" begin
    @testset "$(name)" for (name, m) in lmms
        @test stderror(m, Vcov.simple()) ≈ stderror(m) nans = true
        @testset "$(est)" for est in (Vcov.robust(), Vcov.cluster(:subj))
            se = stderror(m, est)
            @test length(se) == length(coef(m))
            @test isequal(isnan.(se), isnan.(stderror(m)))
            @test se ≈ sqrt.(diag(vcov(m, est))) nans = true
        end
    end
end

@testset "SimpleCovariance" begin
    dc = SimpleCovariance()
    @testset "$(name)" for (name, m) in lmms
        @test isapprox(vcov(m, dc), vcov(m); nans=true)
        @test isapprox(vcov(m, dc; corr=true), vcov(m; corr=true); nans=true)
        @test isapprox(stderror(m, dc), stderror(m); nans=true)
        @test isequal(coeftable(m, dc).cols, coeftable(m).cols)
    end
    # the bias correction is not defined for mixed models
    corrected = SimpleCovariance(; corrected=true)
    @testset "corrected=true throws for $(name)" for (name, m) in (("lmm", fm), ("glmm", gm))
        @test_throws ArgumentError vcov(m, corrected)
        @test_throws ArgumentError vcov(m, corrected; corr=true)
        @test_throws ArgumentError stderror(m, corrected)
        @test_throws ArgumentError coeftable(m, corrected)
        @test_throws ArgumentError sprint(show, m, corrected)
    end
    # GLMMs use the default estimator
    @test vcov(gm, dc) == vcov(gm)
    @test stderror(gm, dc) ≈ stderror(gm)
    @test stderror(gm) ≈ sqrt.(diag(vcov(gm)))
    @test coeftable(gm, dc).cols == coeftable(gm).cols
end

@testset "covariance estimator argument for coeftable and show" begin
    est = Vcov.cluster(:subj)
    dc = SimpleCovariance()
    se = stderror(fm, est)

    @testset "coeftable" begin
        ct = coeftable(fm, est)
        @test ct.cols[2] ≈ se
        @test ct.cols[3] ≈ coef(fm) ./ se
        @test coeftable(fm_rankdef, est).cols[2] ≈ stderror(fm_rankdef, est) nans = true
    end

    @testset "text/plain" begin
        out = sprint(show, MIME("text/plain"), fm; context=:compact => false)
        out_est = sprint(show, MIME("text/plain"), fm, est)
        @test !occursin("Standard errors:", out)
        est_line = "Standard errors: " * sprint(show, MIME("text/plain"), est)
        @test occursin(est_line, out_est)
        ct_est = sprint(show, MIME("text/plain"), coeftable(fm, est))
        @test occursin(ct_est, out_est)
        # the estimator is described after the coefficient table
        @test last(findfirst(ct_est, out_est)) < first(findfirst(est_line, out_est))
        # 2-argument show forwards the estimator
        @test sprint(show, fm, est) == out_est
        # SimpleCovariance is the default
        @test sprint(show, MIME("text/plain"), fm, dc) == out
        @test sprint(show, gm, dc) == sprint(show, gm)
    end

    @testset "$(mime)" for mime in
                           ("text/markdown", "text/html", "text/latex", "text/xelatex")
        out = sprint(show, MIME(mime), fm)
        out_est = sprint(show, MIME(mime), fm, est)
        @test out != out_est
        @test all(s -> occursin(MixedModels.Ryu.writefixed(s, 4), out_est), se)
        @test sprint(show, MIME(mime), fm, dc) == out
    end
end

@testset "return type and symmetry" begin
    @testset "$(name), $(est)" for (name, m) in lmms,
        est in (Vcov.robust(), Vcov.cluster(:subj))

        vc = vcov(m, est)
        # a plain Matrix in both the full-rank and the rank-deficient case
        @test vc isa Matrix{Float64}
        @test isequal(vc, vc')
    end
end

@testset "clubSandwich reference values" begin
    # Reference values from R 4.6.2, lme4 2.1.0 and clubSandwich 0.7.0:
    #
    #   ss <- sleepstudy[seq_len(nrow(sleepstudy)) %% 7 != 0, ]
    #   ox <- oxide[seq_len(nrow(oxide)) %% 5 != 0, ]
    #   f <- lmer(reaction ~ 1 + days + (1 + days | subj), ss, REML = FALSE)
    #   getME(f, "theta")
    #   vcovCR(f, cluster = ss$subj, type = "CR1S")
    #   # similarly with REML = TRUE, with (1 + days || subj), and for
    #   # lmer(Thickness ~ 1 + Source + (1 | Lot / Wafer), ox, REML = FALSE) with cluster = ox$Lot
    #
    # CR1S, with small-sample factor G(N-1)/((G-1)(N-p)), is the CR1 used by Vcov.jl.
    # The subsets are unbalanced on purpose: for these balanced designs, V⁻¹X_g = X_g A
    # for a fixed A, so the CR sandwich does not depend on V and would not test the GLS
    # bread and scores.
    # The Julia models are evaluated at lme4's θ, so that the comparison tests the
    # estimator rather than small differences between the optimizers.
    ss = DataFrame(dataset(:sleepstudy))
    ss = ss[mod.(1:nrow(ss), 7) .!= 0, :]
    ox = DataFrame(dataset(:oxide))
    ox = ox[mod.(1:nrow(ox), 5) .!= 0, :]

    cases = [
        (; name="sleepstudy, ML",
            form=@formula(reaction ~ 1 + days + (1 + days | subj)), data=ss, REML=false,
            cluster=:subj, θ=[0.7377423387, 0.04696588215, 0.2089935453],
            CR1S=[40.68992492352 -1.263647684156; -1.263647684156 2.572859974538]),
        (; name="sleepstudy, REML",
            form=@formula(reaction ~ 1 + days + (1 + days | subj)), data=ss, REML=true,
            cluster=:subj, θ=[0.7747066354, 0.04293016162, 0.2177891632],
            CR1S=[40.69261219182 -1.249981904967; -1.249981904967 2.567601587591]),
        (; name="sleepstudy, zerocorr, ML",
            form=@formula(reaction ~ 1 + days + zerocorr(1 + days | subj)), data=ss,
            REML=false, cluster=:subj, θ=[0.7840962157, 0.2243579148],
            CR1S=[40.53586706748 -1.223999121775; -1.223999121775 2.562502191275]),
        (; name="oxide, nested, ML",
            form=@formula(Thickness ~ 1 + Source + (1 | Lot / Wafer)), data=ox, REML=false,
            cluster=:Lot, θ=[1.888490209, 2.602098464],
            CR1S=[6.804351475105 -6.804351475105; -6.804351475105 53.91648850924]),
    ]

    @testset "$(c.name)" for c in cases
        m = LinearMixedModel(c.form, c.data)
        m.optsum.REML = c.REML
        # evaluate the model at lme4's optimum
        MixedModels.objective!(m, c.θ)
        @test vcov(m, Vcov.cluster(c.cluster)) ≈ c.CR1S rtol = 1e-8
        @test stderror(m, Vcov.cluster(c.cluster)) ≈ sqrt.(diag(c.CR1S)) rtol = 1e-8
    end
end
