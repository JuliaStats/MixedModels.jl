using LinearAlgebra, MixedModels, SparseArrays, Test
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
