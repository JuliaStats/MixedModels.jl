using DataFrames, LinearAlgebra, MixedModels, Random, SparseArrays, StatsModels, Test

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
        @test maximum(abs.(ab - aab)) < 100*eps()
        @test a*bs' == ab
        @test as'*b == ab
        @test as'*bs' == ab
        f = Diagonal(rand(n))
        @test Array(a*f) == Array(a)*f
        @test Array(f*b) == f*Array(b)
    end
end

@testset "reweight!" begin
    rng = Random.MersenneTwister(1234321)
    df = (Y = randn(rng, 400), A = repeat(['N','Y'], outer=200),
        G = repeat('A':'T', inner = 2, outer=10), H = repeat('a':'j', inner=40))
    lmm1 = fit!(LinearMixedModel(@formula(Y ~ 1+A+(1+A|G)+(1+A|H)), df,
        wts  = ones(400)))
    @test loglikelihood(lmm1) ≈ -578.9080978272708
    MixedModels.reweight!(lmm1, ones(400))
    @test loglikelihood(fit!(lmm1)) ≈ -578.9080978272708
end

@testset "lmulλ!" begin
    gendata(n::Int, ng::Int) = gendata(MersenneTwister(42), n, ng)

    function gendata(rng::AbstractRNG, n::Int, ng::Int)
        df = DataFrame(Y = randn(rng, n),
                       X = rand(rng, n),
                       G = rand(rng, 1:ng, n),
                       H = rand(rng, 1:ng, n))
        categorical!(df, [:G, :H])
    end

    @testset "Adjoint{T,ReMat{T,1}}, BlockedSparse{T,1,2}" begin
        # this is an indirect test of lmulΛ! for a blocking structure found in
        # an example in MixedModels.jl#123
        df = gendata(10000, 500)
        f = @formula(Y ~ (1 + X|H) + (1|G))
        m500 = fit!(LinearMixedModel(f, df))
        # the real test here isn't in the theta comparison but in that the fit
        # completes successfully
        @test m500.theta ≈ [0.07797345952252123, -0.17640571417349787, 0, 0]
    end
end
