using LinearAlgebra, MixedModels, Random, SparseArrays, StatsModels, Test
using MixedModels: mulαβ!

@testset "mulαβ!" begin
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
        @test aab ≈ mulαβ!(c, a, bs', true, true)
        @test aab ≈ mulαβ!(c, a, bs')
        @test aab ≈ mulαβ!(c, a, arbt')
        @test aab ≈ mulαβ!(c, a, arbt')
        @test aab ≈ mulαβ!(fill!(c, 0.0), a, arbt', true, true)
        @test maximum(abs, mulαβ!(c, a, arbt', -1.0, true)) ≤ sqrt(eps())
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
        weights = ones(400)))
    @test loglikelihood(lmm1) ≈ -578.9080978272708
    MixedModels.reweight!(lmm1, ones(400))
    @test loglikelihood(fit!(lmm1)) ≈ -578.9080978272708
end
