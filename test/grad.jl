using CategoricalArrays
using FiniteDiff
using ForwardDiff
using LinearAlgebra
using SparseArrays
using StableRNGs
using MixedModels
using Test

using MixedModels:
    GradientWorkspace, GRAD_PANEL, _crossacc_blas3!, _use_blas3_cross,
    dataset

include("modelcache.jl")

# maximum absolute difference between the analytic gradient and the ForwardDiff
# gradient of the objective at θ; restores the model to its fitted state afterwards
function grad_absdiff(m::LinearMixedModel{T}, θ::AbstractVector{T}) where {T}
    g = similar(θ)
    val = objective_gradient!(g, m, θ)
    @test val ≈ objective!(m, θ)
    gfd = ForwardDiff.gradient(m, θ)
    updateL!(setθ!(m, m.optsum.final))
    return maximum(abs, g .- gfd)
end

# a deterministic, non-optimal parameter value in the interior of the parameter space
perturb(θ::AbstractVector) = θ .* 0.75 .+ 0.125

@testset "gradient vs ForwardDiff" begin
    @testset "$(dsnm)[$i]" for dsnm in
                               (:dyestuff, :pastes, :penicillin, :sleepstudy, :kb07),
        (i, m) in enumerate(models(dsnm))

        θ = m.optsum.initial
        tol = 1e-5 * max(1, norm(objective(m)))
        @test grad_absdiff(m, θ) < tol
        @test grad_absdiff(m, perturb(θ)) < tol
        @test grad_absdiff(m, m.optsum.final) < tol
    end

    @testset "near-zero gradient at interior optimum" begin
        for m in (only(models(:dyestuff)), last(models(:sleepstudy)))
            g = similar(m.optsum.final)
            objective_gradient!(g, m, m.optsum.final)
            # the tolerance reflects how sharply the derivative-free fit converged,
            # not the accuracy of the gradient
            @test norm(g) < 5e-3
            updateL!(setθ!(m, m.optsum.final))
        end
    end

    @testset "REML" begin
        for (f, dsnm) in ((last(fms[:sleepstudy]), :sleepstudy),
            (only(fms[:penicillin]), :penicillin),
            (last(fms[:pastes]), :pastes))
            m = fit(MixedModel, f, dataset(dsnm); REML=true, progress=false)
            θ = perturb(m.optsum.initial)
            g = similar(θ)
            val = objective_gradient!(g, m, θ)
            @test val ≈ objective!(m, θ)
            gfd = ForwardDiff.gradient(m, θ)
            @test g ≈ gfd rtol = 1e-6 atol = 1e-8
        end
    end

    @testset "fixed sigma" begin
        for REML in (false, true)
            m = fit(MixedModel, last(fms[:sleepstudy]), dataset(:sleepstudy);
                σ=25.0, REML, progress=false)
            θ = perturb(m.optsum.initial)
            g = similar(θ)
            val = objective_gradient!(g, m, θ)
            @test val ≈ objective!(m, θ)
            # the ForwardDiff extension holds σ at optsum.sigma when it is fixed,
            # matching the analytic gradient
            gfd = ForwardDiff.gradient(m, θ)
            @test g ≈ gfd rtol = 1e-6 atol = 1e-8
            gff = FiniteDiff.finite_difference_gradient(Base.Fix1(objective!, m), θ)
            @test g ≈ gff rtol = 1e-4 atol = 1e-4
        end
    end

    @testset "workspace reuse and argument checking" begin
        m = last(models(:sleepstudy))
        θ = perturb(m.optsum.initial)
        w = GradientWorkspace(m)
        g1 = similar(θ)
        g2 = similar(θ)
        objective_gradient!(w, g1, updateL!(setθ!(m, θ)))
        objective_gradient!(w, g2, m)   # reusing the workspace must be idempotent
        @test g1 == g2
        @test_throws DimensionMismatch objective_gradient!(similar(θ, 2), m)
        updateL!(setθ!(m, m.optsum.final))
    end

    @testset "BLAS-3 cross-term kernel" begin
        # the panelled kernel must equal the dense reference ⟨A, Xrr' Xrb⟩, and must
        # span more than one panel to exercise the panel-boundary bookkeeping
        rng = StableRNG(1234)
        qr, qb = 40, 3 * GRAD_PANEL + 7
        Xrr = randn(rng, qr, qr)
        Xrb = randn(rng, qr, qb)
        A = sprand(rng, qr, qb, 0.2)
        Pp = Matrix{Float64}(undef, qr, GRAD_PANEL)
        ref = sum(
            A[u, v] * dot(view(Xrr, :, u), view(Xrb, :, v)) for
            (u, v, _) in zip(findnz(A)...)
        )
        @test _crossacc_blas3!(Pp, A, Xrr, Xrb) ≈ ref
    end

    @testset "BLAS-3 cross path matches sparse path" begin
        # a small, dense partially-crossed design: sparse A[2,1] but dense Cholesky fill,
        # dense enough to take the gated BLAS-3 path
        # sparse (density ≈ 0.06) so A[2,1] is not densified, yet above the BLAS-3 gate
        rng = StableRNG(42)
        n, ng, nh = 1200, 150, 120
        tbl = (; y=randn(rng, n),
            g=categorical(rand(rng, 1:ng, n)), h=categorical(rand(rng, 1:nh, n)))
        gcontr = Dict(:g => Grouping(), :h => Grouping())
        m = LinearMixedModel(@formula(y ~ 1 + (1 | g) + (1 | h)), tbl; contrasts=gcontr)
        θ = [0.7, 1.3]
        updateL!(setθ!(m, θ))
        wb = GradientWorkspace(m)                 # gate active
        ws = GradientWorkspace(m)
        ws = GradientWorkspace(ws.X, ws.S, ws.C1, ws.C2, ws.G, Matrix{Float64}(undef, 0, 0))
        @test _use_blas3_cross(wb, m, 2, 1)       # dense crossing → BLAS-3
        @test !_use_blas3_cross(ws, m, 2, 1)
        gb = zeros(2)
        gs = zeros(2)
        objective_gradient!(wb, gb, m)
        objective_gradient!(ws, gs, m)
        @test gb ≈ gs rtol = 1e-10                # same math up to BLAS-3 vs BLAS-1 reassociation
        @test gb ≈ ForwardDiff.gradient(m, θ) rtol = 1e-7
    end
end

@testset "gradient-based optimization" begin
    @testset "LD_LBFGS $(dsnm)" for (dsnm, f) in
                                    (
        (:sleepstudy, last(fms[:sleepstudy])), (:penicillin, only(fms[:penicillin]))
    )
        mref = fit(MixedModel, f, dataset(dsnm); progress=false)
        m = fit(MixedModel, f, dataset(dsnm); optimizer=:LD_LBFGS, progress=false)
        @test m.optsum.optimizer == :LD_LBFGS
        @test m.optsum.returnvalue in (:SUCCESS, :FTOL_REACHED, :XTOL_REACHED)
        @test m.optsum.fmin ≈ mref.optsum.fmin atol = 1e-6
        @test m.θ ≈ mref.θ atol = 1e-3
    end

    @test :LD_LBFGS in MixedModels.optimizers(Val(:nlopt))

    @testset "gradient=:forwarddiff" begin
        m = last(models(:sleepstudy))
        θ = perturb(m.optsum.initial)
        w = MixedModels.fd_gradient_workspace(m)
        g_fd = similar(θ)
        val_fd = MixedModels.fd_objective_gradient!(w, g_fd, m, θ)
        g = similar(θ)
        val = objective_gradient!(g, m, θ)
        @test val_fd ≈ val
        @test g_fd ≈ g rtol = 1e-8 atol = 1e-10
        # repeated evaluation with the cached workspace is idempotent
        g2 = similar(θ)
        @test MixedModels.fd_objective_gradient!(w, g2, m, θ) == val_fd
        @test g2 == g_fd
        @test_throws ArgumentError MixedModels.fd_objective_gradient!(
            w, g2, only(models(:penicillin)), θ)
        updateL!(setθ!(m, m.optsum.final))

        mref = fit(MixedModel, last(fms[:sleepstudy]), dataset(:sleepstudy);
            optimizer=:LD_LBFGS, progress=false)
        mfd = fit(MixedModel, last(fms[:sleepstudy]), dataset(:sleepstudy);
            optimizer=:LD_LBFGS, gradient=:forwarddiff, progress=false)
        @test mfd.optsum.gradient == :forwarddiff
        @test mfd.optsum.returnvalue in (:SUCCESS, :FTOL_REACHED, :XTOL_REACHED)
        @test mfd.optsum.fmin ≈ mref.optsum.fmin atol = 1e-6
        @test mfd.θ ≈ mref.θ atol = 1e-3

        @test_throws ArgumentError fit(MixedModel, last(fms[:sleepstudy]),
            dataset(:sleepstudy); gradient=:badsource, progress=false)
    end

    @testset "profile after gradient-based fit" begin
        # the profiling objectives are derivative-free; profiling a model fitted
        # with an LD optimizer must fall back to a derivative-free optimizer
        m = fit(MixedModel, first(fms[:sleepstudy]), dataset(:sleepstudy);
            optimizer=:LD_LBFGS, progress=false)
        @test profile(m) isa MixedModelProfile
    end
end
