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
    _mulsub!, _ldivL!, _patternmirror, _productcontained, _blockaligned,
    GRAD_STATIC_K, GRAD_STATIC_ROWS, ReMat, TriangularRFP, UniformBlockDiagonal,
    _facecontract!, _facecontract_rows!, _gramfaces_acc!, _selcontract!,
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

    @testset "crossed designs $(label) $(REML ? "REML" : "ML")" for (label, f) in (
            ("scalar-crossed", first(fms[:kb07])),
            ("vector-crossed", last(fms[:kb07]))),
        REML in (false, true)
        # subject × item crossing exercises the sparse/BLAS-3 off-diagonal path
        # (scalar terms) and the dense _densepair! path (vector terms); the REML
        # cases are the capability the earlier ML-only prototype could not handle
        m = fit(MixedModel, f, dataset(:kb07); REML, progress=false)
        θ = perturb(m.optsum.initial)
        g = similar(θ)
        val = objective_gradient!(g, m, θ)
        @test val ≈ objective!(m, θ)
        gfd = ForwardDiff.gradient(m, θ)
        @test g ≈ gfd rtol = 1e-6 atol = 1e-6
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

    @testset "nested grouping factors (sparse workspace)" begin
        # nested designs keep the off-diagonal L blocks sparse and the diagonal
        # blocks block-diagonal; the workspace must mirror that structure instead
        # of allocating dense blocks of L⁻¹ (issue: gradient-based fits of models
        # like fggk21 ran out of memory)
        rng = StableRNG(20260711)
        n = 6000
        ng, nh = 25, 6
        g = rand(rng, 1:ng, n)
        h = (g .- 1) .* nh .+ rand(rng, 1:nh, n)    # h nested in g
        x = randn(rng, n)
        tbl = (; y=randn(rng, n), x, g=categorical(g), h=categorical(h))

        @testset "$(label)" for (label, f, sptype) in (
            ("vector-vector", @formula(y ~ 1 + x + (1 + x | g) + (1 + x | h)),
                UniformBlockDiagonal),
            ("scalar-under-vector", @formula(y ~ 1 + x + (1 + x | g) + (1 | h)),
                UniformBlockDiagonal),
            ("vector-under-scalar", @formula(y ~ 1 + x + (1 | g) + (1 + x | h)),
                Diagonal),
        )
            m = LinearMixedModel(f, tbl)
            w = GradientWorkspace(m)
            # the big diagonal block of X = L⁻¹ mirrors L[1,1], the fill block
            # X[2,1] mirrors the nested sparsity of L[2,1], and the pair buffer
            # S[2,1] holds only the entries of S on the pattern of A[2,1]
            @test !isa(w.X[1, 1], Matrix)
            @test w.X[2, 1] isa SparseMatrixCSC
            @test w.X[2, 2] isa sptype
            @test w.S[2, 1] isa SparseMatrixCSC
            θ = perturb(m.optsum.initial)
            gan = similar(θ)
            val = objective_gradient!(w, gan, updateL!(setθ!(m, θ)))
            @test val ≈ objective!(m, θ)
            @test gan ≈ ForwardDiff.gradient(m, θ) rtol = 1e-6 atol = 1e-8
        end

        @testset "three-level scalar nesting" begin
            na, nb, nc = 15, 4, 3
            a = rand(rng, 1:na, n)
            b = (a .- 1) .* nb .+ rand(rng, 1:nb, n)
            c = (b .- 1) .* nc .+ rand(rng, 1:nc, n)
            tbl3 = (; y=randn(rng, n), x,
                a=categorical(a), b=categorical(b), c=categorical(c))
            # amalgamate=false keeps the three nested scalar terms separate, so the
            # sparse block column of X has an intermediate term (the sparse-sparse
            # `_mulsub!` accumulation and its pattern-containment check)
            m = LinearMixedModel(
                @formula(y ~ 1 + x + (1 | a) + (1 | b) + (1 | c)), tbl3;
                amalgamate=false)
            w = GradientWorkspace(m)
            @test w.X[2, 1] isa SparseMatrixCSC
            @test w.X[3, 1] isa SparseMatrixCSC
            @test w.X[3, 2] isa SparseMatrixCSC
            θ = perturb(m.optsum.initial)
            gan = similar(θ)
            objective_gradient!(w, gan, updateL!(setθ!(m, θ)))
            @test gan ≈ ForwardDiff.gradient(m, θ) rtol = 1e-6 atol = 1e-8
        end

        @testset "nested vector + crossed zerocorr (fggk21 structure) $(REML ? "REML" : "ML")" for REML in
                                                                                                   (
            false, true
        )
            co = rand(rng, 1:4, n)
            tblf = (; y=randn(rng, n), x,
                g=categorical(g), h=categorical(h), co=categorical(co))
            m = LinearMixedModel(
                @formula(y ~ 1 + x + (1 + x | g) + (1 + x | h) + zerocorr(1 + x | co)),
                tblf)
            m.optsum.REML = REML
            w = GradientWorkspace(m)
            @test w.X[1, 1] isa UniformBlockDiagonal
            @test w.X[2, 1] isa SparseMatrixCSC
            @test w.S[2, 1] isa SparseMatrixCSC
            θ = perturb(m.optsum.initial)
            gan = similar(θ)
            val = objective_gradient!(w, gan, updateL!(setθ!(m, θ)))
            @test val ≈ objective!(m, θ)
            @test gan ≈ ForwardDiff.gradient(m, θ) rtol = 1e-6 atol = 1e-8
        end
    end

    @testset "sparse workspace kernels" begin
        # kernels for sparse mirrors of L blocks, checked against dense references;
        # some type combinations arise only in designs (deep nesting under a dense
        # diagonal block) that model fits rarely reach
        rng = StableRNG(97)
        kr, kc = 3, 2
        nlr, nlc = 4, 6
        # block-aligned sparse A: one kr×kc block per column block
        rowblk = rand(rng, 0:(nlr - 1), nlc)
        rows = reduce(vcat, [rowblk[j] * kr .+ (1:kr) for j in 1:nlc for _ in 1:kc])
        cols = reduce(vcat, [fill((j - 1) * kc + l, kr) for j in 1:nlc for l in 1:kc])
        A = SparseMatrixCSC{Float64,Int32}(
            sparse(rows, cols, randn(rng, length(rows)), nlr * kr, nlc * kc)
        )
        @test _blockaligned(A, kr, kc)
        @test !_blockaligned(A, kr + 1, kc)

        # C -= A * D and C -= A * U on the pattern of A (block-diagonal right factor)
        D = Diagonal(randn(rng, nlc * kc))
        U = UniformBlockDiagonal(randn(rng, kc, kc, nlc))
        for B in (D, U)
            C = _patternmirror(A)
            fill!(nonzeros(C), 0.0)
            _mulsub!(C, A, B)
            @test Matrix(C) ≈ -Matrix(A) * Matrix(B)
        end

        # sparse-sparse scatter: pattern(A * B) ⊆ pattern(C)
        B = SparseMatrixCSC{Float64,Int32}(
            sparse(
                reduce(vcat, [(j - 1) * kc .+ (1:kc) for j in 1:nlc]),
                reduce(vcat, [fill(j, kc) for j in 1:nlc]),
                randn(rng, nlc * kc),
                nlc * kc, nlc,
            ),
        )
        Cpat = SparseMatrixCSC{Float64,Int32}(A * B)   # the exact product pattern
        @test _productcontained(A, B, Cpat)
        C = _patternmirror(Cpat)
        fill!(nonzeros(C), 0.0)
        _mulsub!(C, A, B)
        @test Matrix(C) ≈ -Matrix(A) * Matrix(B)
        # containment must fail for a pattern missing a product entry
        Cmiss = SparseMatrixCSC{Float64,Int32}(
            sparse(
                rowvals(Cpat)[2:end],
                reduce(vcat, [fill(v, length(nzrange(Cpat, v))) for v in axes(Cpat, 2)])[2:end],
                nonzeros(Cpat)[2:end],
                size(Cpat)...,
            ),
        )
        @test !_productcontained(A, B, Cmiss)

        # dense C -= A * B with both factors sparse
        Cd = randn(rng, size(A, 1), size(B, 2))
        ref = Cd - Matrix(A) * Matrix(B)
        _mulsub!(Cd, A, B)
        @test Cd ≈ ref

        # block-diagonal solve on a sparse right-hand side
        Ld = randn(rng, kr, kr, nlr)
        for f in 1:nlr    # make the faces well-conditioned lower triangles
            Ld[:, :, f] = LowerTriangular(view(Ld, :, :, f)) + 3 * I(kr)
        end
        Ljj = UniformBlockDiagonal(Ld)
        X = _patternmirror(A)
        copyto!(nonzeros(X), nonzeros(A))
        _ldivL!(Ljj, X)
        @test Matrix(X) ≈ LowerTriangular(Matrix(Ljj)) \ Matrix(A)
    end

    @testset "statically sized face and block kernels" begin
        # The face and block loops are unrolled into register arithmetic for term
        # dimensions up to GRAD_STATIC_K and fall back to `mul!` past it, so every kernel
        # is checked against a dense reference on both sides of the cutoff.  `_facecontract!`
        # additionally switches on the per-face row count at GRAD_STATIC_ROWS.
        rng = StableRNG(20260810)
        Ks = (1, 2, GRAD_STATIC_K, GRAD_STATIC_K + 1)

        # only λ and the term dimension are read by these kernels; the rest of the ReMat
        # is filler
        function testremat(::Val{K}, nlev::Int) where {K}
            λ = LowerTriangular(tril(randn(rng, K, K)) + 2I)
            return ReMat{Float64,K}(
                nothing, Int32[1], 1:nlev, ["c$i" for i in 1:K],
                zeros(K, 1), zeros(K, 1), λ, collect(1:(K * K)),
                spzeros(Float64, Int32, nlev * K, 1), zeros(K, 1),
            )
        end

        @testset "_selcontract! kr=$kr kb=$kb" for kr in Ks, kb in Ks
            nlr, nlc = 5, 7
            # block-aligned sparse A: one kr×kc block per column block
            rowblk = rand(rng, 0:(nlr - 1), nlc)
            rows = reduce(vcat, [rowblk[j] * kr .+ (1:kr) for j in 1:nlc for _ in 1:kb])
            cols = reduce(vcat, [fill((j - 1) * kb + l, kr) for j in 1:nlc for l in 1:kb])
            A = sparse(rows, cols, randn(rng, length(rows)), nlr * kr, nlc * kb)
            @test _blockaligned(A, kr, kb)
            Sp = _patternmirror(A)
            copyto!(nonzeros(Sp), randn(rng, nnz(A)))

            rtr = testremat(Val(kr), nlr)
            rtb = testremat(Val(kb), nlc)
            Gb = zeros(kb, kb)
            Gr = zeros(kr, kr)
            _selcontract!(Gb, Gr, A, Sp, rtr, rtb)

            # reference: sum over the kr×kb blocks of the dense A and Sp.  Blocks outside
            # the pattern are zero in both, so summing over all of them is equivalent.
            Ad, Sd = Matrix(A), Matrix(Sp)
            Gbref = zeros(kb, kb)
            Grref = zeros(kr, kr)
            for I in 1:nlr, J in 1:nlc
                Ablk = Ad[((I - 1) * kr + 1):(I * kr), ((J - 1) * kb + 1):(J * kb)]
                Sblk = Sd[((I - 1) * kr + 1):(I * kr), ((J - 1) * kb + 1):(J * kb)]
                Gbref += Ablk' * (rtr.λ * Sblk)
                Grref += (Ablk * rtb.λ) * Sblk'
            end
            @test Gb ≈ Gbref
            @test Gr ≈ Grref
        end

        @testset "_gramfaces_acc! K=$K" for K in Ks
            nf = 6
            q = 9
            dense = randn(rng, q, nf * K)
            ubd = UniformBlockDiagonal(randn(rng, K, K, nf))
            # block-aligned sparse X: two complete K-column blocks of nonzeros per face
            srows = reduce(
                vcat, [(2 * K) .* (i - 1) .+ (1:(2 * K)) for i in 1:nf for _ in 1:K]
            )
            scols = reduce(vcat, [fill((i - 1) * K + l, 2 * K) for i in 1:nf for l in 1:K])
            spx = sparse(srows, scols, randn(rng, length(srows)), 2 * K * nf, K * nf)
            rfp = TriangularRFP(collect(LowerTriangular(randn(rng, K * nf, K * nf))), :L)
            for X in (dense, ubd, spx, rfp)
                datstatic = zeros(K, K, nf)
                datgeneric = zeros(K, K, nf)
                _gramfaces_acc!(datstatic, X, Val(K))
                _gramfaces_acc!(datgeneric, X)
                @test datstatic ≈ datgeneric
                Xd = Matrix(X)
                for f in 1:nf
                    Xv = view(Xd, :, ((f - 1) * K + 1):(f * K))
                    @test view(datstatic, :, :, f) ≈ Xv'Xv
                end
            end
        end

        # rows per face straddling GRAD_STATIC_ROWS, where `_facecontract!` switches paths
        @testset "_facecontract! K=$K q=$q" for K in Ks,
            q in (GRAD_STATIC_ROWS ÷ 2, 2 * GRAD_STATIC_ROWS)

            nf = 5
            rt = testremat(Val(K), nf)
            C = randn(rng, q, nf * K)
            S = randn(rng, q, nf * K)
            @test _facecontract!(zeros(K, K), C, S, rt) ≈
                _facecontract!(zeros(K, K), C, S)
            # and against the definition
            ref = sum(
                view(C, :, ((f - 1) * K + 1):(f * K))' *
                view(S, :, ((f - 1) * K + 1):(f * K))
                for f in 1:nf
            )
            @test _facecontract!(zeros(K, K), C, S, rt) ≈ ref
            # `_facecontract_rows!` transposes the roles of faces and columns
            Cr = randn(rng, nf * K, q)
            Sr = randn(rng, nf * K, q)
            @test _facecontract_rows!(zeros(K, K), Cr, Sr, rt) ≈
                _facecontract_rows!(zeros(K, K), Cr, Sr)
            refr = sum(
                view(Cr, ((f - 1) * K + 1):(f * K), :) *
                view(Sr, ((f - 1) * K + 1):(f * K), :)'
                for f in 1:nf
            )
            @test _facecontract_rows!(zeros(K, K), Cr, Sr, rt) ≈ refr
        end
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
        ws = GradientWorkspace(
            ws.X, ws.S, ws.C1, ws.C2, ws.G, Matrix{Float64}(undef, 0, 0), ws.path
        )
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
