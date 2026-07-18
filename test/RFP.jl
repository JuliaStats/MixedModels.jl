using ForwardDiff
using LinearAlgebra
using MixedModels
using StableRNGs
using Test
using SparseArrays

using MixedModels: GradientWorkspace, GRAD_PANEL, HermitianRFP, TriangularRFP,
    _colsumabs2!, _crossacc_blas3!, _gramacc!, _rfpviews, _xdot, _xdotRFP,
    dataset, rankUpdate!

# tests of the rankUpdate! method when `typeof(C)` is `HermitianRFP`
@testset "rankUpHermitianRFP" begin
    A9 = float(sprand(Bool, 9, 12, 0.3))
    T = TriangularRFP(collect(A9 * A9'), :L)
    C9 = HermitianRFP(T.data, T.transr, T.uplo)
    rankUpdate!(C9, A9, -1.0, 1.0)
    @test all(iszero, C9.data)
    A8 = float(sprand(Bool, 8, 12, 0.3))
    T = TriangularRFP(collect(A8 * A8'), :L)
    C8 = HermitianRFP(T.data, T.transr, T.uplo)
    rankUpdate!(C8, A8, -1.0, 1.0)
    @test all(iszero, C8.data)
end

# the hand-written kernels on the packed storage regions of a TriangularRFP block
# of the gradient workspace, checked against dense references for even and odd
# orders (the two storage layouts)
@testset "packed layout kernels" begin
    rng = StableRNG(314)
    @testset "order $n" for n in (1, 2, 5, 8, 9)
        Xd = Matrix(LowerTriangular(randn(rng, n, n)))
        X = TriangularRFP(copy(Xd), :L)
        tv, uv, m1, m2 = _rfpviews(X)
        @test all(tv[i, j] == Xd[i, j] for j in 1:m1 for i in j:n)
        @test all(uv[j - m1, i - m1] == Xd[i, j] for j in (m1 + 1):n for i in j:n)

        d = zeros(n)
        @test _colsumabs2!(d, X) ≈ vec(sum(abs2, Xd; dims=1))

        Xb = randn(rng, n, 4)
        @test all(
            _xdot(X, Xb, u, v) ≈ dot(view(Xd, :, u), view(Xb, :, v)) for
            u in 1:n, v in 1:4
        )
        @test all(
            _xdotRFP(X, a, b) ≈ dot(view(Xd, :, a), view(Xd, :, b)) for
            a in 1:n, b in 1:n
        )

        S = randn(rng, n, 4)
        @test _gramacc!(copy(S), X, Xb) ≈ S + Xd'Xb

        # the panelled cross-term kernel, spanning more than one column panel
        qb = 2 * GRAD_PANEL + 7
        Yb = randn(rng, n, qb)
        A = sparse(randn(rng, n, qb))
        Pp = Matrix{Float64}(undef, n, GRAD_PANEL)
        ref = sum(
            A[u, v] * dot(view(Xd, :, u), view(Yb, :, v)) for
            (u, v, _) in zip(findnz(A)...)
        )
        @test _crossacc_blas3!(Pp, A, X, Yb) ≈ ref
    end
end

# a deterministic, non-optimal parameter value in the interior of the parameter space
rfpperturb(θ::AbstractVector) = θ .* 0.75 .+ 0.125

# compare the objective, the analytic gradient, and the ForwardDiff gradient of a
# model whose fill-in diagonal blocks are forced into packed (RFP) storage against
# the same model in the default dense storage
function rfp_gradcheck(f, tbl; REML=false)
    m0 = LinearMixedModel(f, tbl)
    mR = LinearMixedModel(f, tbl; RFPthreshold=1)
    m0.optsum.REML = mR.optsum.REML = REML
    @test any(Base.Fix2(isa, TriangularRFP), mR.L)
    θ = rfpperturb(mR.optsum.initial)
    g0 = similar(θ)
    gR = similar(θ)
    @test objective_gradient!(gR, mR, θ) ≈ objective_gradient!(g0, m0, θ)
    @test gR ≈ g0 rtol = 1e-8 atol = 1e-10
    # the ForwardDiff extension expands the packed blocks to dense lower triangles
    @test gR ≈ ForwardDiff.gradient(mR, θ) rtol = 1e-6 atol = 1e-8
    return mR
end

@testset "gradient with RFP storage" begin
    pen = dataset(:penicillin)
    fpen = @formula(diameter ~ 1 + (1 | plate) + (1 | sample))

    # crossed scalar factors, even-order packed block (BLAS-3 cross-term path)
    mR = rfp_gradcheck(fpen, pen)
    w = GradientWorkspace(mR)
    @test w.X[2, 2] isa TriangularRFP   # the packed inverse mirrors the L block

    rng = StableRNG(20260718)
    n = 600
    tbl = (;
        y=randn(rng, n),
        x=randn(rng, n),
        g1=rand(rng, string.('A':'M'), n),
        g2=rand(rng, string.(1:7), n),
        g3=rand(rng, string.(1:9), n),
    )
    # odd-order packed block
    rfp_gradcheck(@formula(y ~ 1 + x + (1 | g1) + (1 | g2)), tbl)
    # two packed blocks, with a dense fill block below the first of them
    rfp_gradcheck(@formula(y ~ 1 + x + (1 | g1) + (1 | g2) + (1 | g3)), tbl)
    # crossed vector-valued terms (UniformBlockDiagonal A blocks)
    rfp_gradcheck(
        @formula(rt_trunc ~ 1 + prec + (1 + prec | subj) + (1 + prec | item)),
        dataset(:kb07),
    )
    # REML weighting of the fixed-effects rows
    rfp_gradcheck(fpen, pen; REML=true)

    @testset "gradient-based fit" begin
        mref = fit(MixedModel, fpen, pen; progress=false)
        mlb = fit(MixedModel, fpen, pen;
            RFPthreshold=1, optimizer=:LD_LBFGS, progress=false)
        @test any(Base.Fix2(isa, TriangularRFP), mlb.L)
        @test mlb.optsum.returnvalue in (:SUCCESS, :FTOL_REACHED, :XTOL_REACHED)
        @test mlb.optsum.fmin ≈ mref.optsum.fmin atol = 1e-6
        @test mlb.θ ≈ mref.θ atol = 1e-3
    end
end
