using LinearAlgebra
using MixedModels
using PooledArrays
using Random
using SparseArrays
using Test

using MixedModels: rankUpdate!, BlockedSparse, HermitianRFP, TriangularRFP

@testset "mul!" begin
    for (m, p, n, q, k) in (
        (10, 0.7, 5, 0.3, 15),
        (100, 0.01, 100, 0.01, 20),
        (100, 0.1, 100, 0.2, 100),
    )
        a = sprand(m, n, p)
        b = sprand(n, k, q)
        as = sparse(a')
        bs = sparse(b')
        c = zeros(m, k)
        ab = a * b
        arbt = Array(b')
        aab = Array(a) * Array(b)
        @test aab ≈ mul!(c, a, bs', true, true)
        @test aab ≈ mul!(c, a, bs')
        @test aab ≈ mul!(c, a, arbt')
        @test aab ≈ mul!(c, a, arbt')
        @test aab ≈ mul!(fill!(c, 0.0), a, arbt', true, true)
        @test maximum(abs, mul!(c, a, arbt', -1.0, true)) ≤ sqrt(eps())
        @test maximum(abs.(ab - aab)) < 100 * eps()
        @test a * bs' == ab
        @test as' * b == ab
        @test as' * bs' == ab
        f = Diagonal(rand(n))
        @test Array(a * f) == Array(a) * f
        @test Array(f * b) == f * Array(b)
    end
end

@testset "reweight!" begin
    rng = MersenneTwister(1234321)
    df = (
        Y=randn(rng, 400),
        A=repeat(PooledArray(["N", "Y"]); outer=200),
        G=repeat(PooledArray(string.('A':'T')); inner=2, outer=10),
        H=repeat(PooledArray(string.('a':'j')); inner=40),
    )
    m1 = fit(
        MixedModel, @formula(Y ~ 1 + A + (1 + A | G) + (1 + A | H)), df; progress=false
    )
    wm1 = fit(
        MixedModel,
        @formula(Y ~ 1 + A + (1 + A | G) + (1 + A | H)),
        df;
        wts=ones(400),
        progress=false,
    )
    @test loglikelihood(wm1) ≈ loglikelihood(m1)
    MixedModels.reweight!(wm1, ones(400))
    @test loglikelihood(refit!(wm1; progress=false)) ≈ loglikelihood(m1)
end

@testset "rankupdate!" begin
    x = [1 1; 1 1]
    # in Julia 1.6+, typeof(x) == Matrix{Int64}
    # in < 1.6, typeof(x) == Array{Int64, 2}
    err = ErrorException(
        "We haven't implemented a method for $(typeof(x)), $(typeof(x)). Please file an issue on GitHub.",
    )
    @test_throws ErrorException rankUpdate!(x, x, 1, 1)
    L21 = sprand(MersenneTwister(42), 100, 1000, 0.05)
    L22L = rankUpdate!(Symmetric(zeros(100, 100), :L), L21, 1.0, 1.0)
    @test L22L ≈
        rankUpdate!(Symmetric(zeros(100, 100), :U), sparse(transpose(L21)), 1.0, 1.0)
end

@testset "rankUpdate! HermitianRFP" begin
    rng = MersenneTwister(1234)
    rfp(S) = Hermitian(TriangularRFP(Matrix(LowerTriangular(S)), :L), :L)

    # crossed factors ⇒ an off-diagonal L block stored as BlockedSparse
    d3 = MixedModels.dataset(:d3)
    m = fit!(
        LinearMixedModel(@formula(y ~ 1 + u + (1 + u | g) + (1 + u | h) + (1 + u | i)), d3);
        progress=false,
    )
    Ablk = m.L[findfirst(a -> a isa BlockedSparse, m.L)]
    @test Ablk isa BlockedSparse
    Sblk = zeros(size(Ablk, 1), size(Ablk, 1)) + I

    testcases = Any[(Ablk, Sblk)]
    # both parities of n are needed: even n packs into the tall RFP layout, odd n the wide one
    for n in (6, 7)
        # a symmetric, lower-stored base matrix and the packed HermitianRFP holding it
        S = let M = randn(rng, n, n)
            M * M' + n * I
        end
        # dense and (Int32) sparse updates
        push!(testcases, (randn(rng, n, 4), S))
        Asparse = convert(SparseMatrixCSC{Float64,Int32}, sparse(sprand(rng, n, 20, 0.3)))
        push!(testcases, (Asparse, S))
    end

    for (A, base) in testcases
        for (α, β) in ((1.0, 1.0), (-1.0, 1.0), (0.5, 2.0))
            ref = rankUpdate!(Hermitian(Matrix(base), :L), A, α, β)
            got = rankUpdate!(rfp(base), A, α, β)
            @test got isa HermitianRFP
            @test LowerTriangular(Matrix(got)) ≈ LowerTriangular(Matrix(ref))
        end
    end

    # the sparse kernel only handles lower, non-transposed storage of a matching size
    S = let M = randn(rng, 6, 6)
        M * M' + 6I
    end
    upper = Hermitian(TriangularRFP(Matrix(UpperTriangular(S)), :U), :U)
    @test_throws ArgumentError rankUpdate!(upper, sprand(rng, 6, 4, 0.5), 1.0, 1.0)
    @test_throws DimensionMismatch rankUpdate!(rfp(S), sprand(rng, 5, 4, 0.5), 1.0, 1.0)

    # rowval must be sorted within a column, since both inner loops start at indj and so
    # assume i ≥ j.  For n = 6 the packed array has 3 columns, putting row 3 in the
    # trapezoidal branch and row 4 in the triangular one; either way this must be caught.
    for rows in ([3, 1], [4, 2])
        unsorted = SparseMatrixCSC(6, 1, [1, 3], rows, ones(2))
        @test_throws AssertionError rankUpdate!(rfp(S), unsorted, 1.0, 1.0)
    end
end

#=  I don't see this testset as meaningful b/c diagonal A does not occur after amalgamation of ReMat's for the same grouping factor - D.B.
@testset "rankupdate!" begin
    @test ones(2, 2) == rankUpdate!(Hermitian(zeros(2, 2)), ones(2))
    d2 = Diagonal(fill(2., 2))
    @test Diagonal(fill(5.,2)) == rankUpdate!(Diagonal(ones(2)), d2, 1.)
    @test Diagonal(fill(-3.,2)) == rankUpdate!(Diagonal(ones(2)), d2, -1.)

    # when converting straight from diagonal to symmetric, the type is different
    @test Diagonal(fill(5.,2)) == rankUpdate!(Symmetric(Matrix(1. * I(2)), :L), d2)
    # generic method
    @test Diagonal(fill(5.,2)) == rankUpdate!(Matrix(1. * I(2)), d2)
end
=#

@testset "lmulλ!" begin
    levs(ng, tag='S') = string.(tag, lpad.(string.(1:ng), ndigits(ng), '0'))

    function gendata(rng::AbstractRNG, n::Integer, ng::Integer, nh::Integer)
        return (
            Y=randn(rng, n),
            X=rand(rng, n),
            G=PooledArray(rand(rng, levs(ng, 'G'), n)),
            H=PooledArray(rand(rng, levs(nh, 'H'), n)),
        )
    end
    gendata(n::Integer, ng::Integer, nh::Integer) = gendata(MersenneTwister(42), n, ng, nh)
    gendata(n::Integer, ng::Integer) = gendata(MersenneTwister(42), n, ng, ng)

    @testset "Adjoint{T,ReMat{T,1}}, BlockedSparse{T,1,2}" begin
        # this is an indirect test of lmulΛ! for a blocking structure found in
        # an example in MixedModels.jl#123
        df = gendata(10000, 500)
        f = @formula(Y ~ (1 + X | H) + (1 | G))
        m500 = fit!(LinearMixedModel(f, df); progress=false)
        # the real test here isn't in the theta comparison but in that the fit
        # completes successfully
        @test length(m500.u) == 2
    end
end
