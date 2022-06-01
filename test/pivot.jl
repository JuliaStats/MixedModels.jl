using LinearAlgebra, StableRNGs, StatsModels, Test

import MixedModels: statsrank

xtx(X) = Symmetric(X'X, :U)  # creat the symmetric matrix X'X from X
LinearAlgebra.rank(F::LinearAlgebra.QRPivoted; tol=1e-8) = searchsortedlast(abs.(diag(F.R)), tol, rev=true);

const rng = StableRNG(4321234)

const simdat = (
    G = repeat('A':'T', inner=10),
    H = repeat('a':'e', inner=2, outer=20),
    U = repeat(0.:9, outer=20),
    V = repeat(-4.5:4.5, outer=20),
    Y = 0.1 * randn(rng, 200),
    Z = rand(rng, 200)
)

@testset "fullranknumeric" begin
    mm = modelmatrix(@formula(Y ~ 1 + U), simdat)
    r, pivot = statsrank(mm)
    @test pivot == 1:2
end

@testset "fullrankcategorical" begin
    mm = modelmatrix(@formula(Y ~ 1 + G*H), simdat)
    r, pivot = statsrank(mm)
    @test r == 100
    @test pivot == 1:100
end

@testset "dependentcolumn" begin
    mm = modelmatrix(@formula(Y ~ 1 + U + V + Z), simdat)
    r, pivot = statsrank(mm)
    # V is just mean-centered U
    # so either V or U gets pivoted out
    # perm1 = [1,2,4,3] # x86-64 OpenBLAS
    # perm2 = [1,3,4,2] # Apple M1
    @test r == 3
    @test pivot[1] == 1 # intercept remains
    @test pivot[3] == 4 # z doesn't get pivoted
    @test pivot[4] in [2, 3]
end

@testset "qr missing cells" begin
    mm = modelmatrix(@formula(Y ~ 1 + G*H), simdat)[5:end,:]
    r, pivot = statsrank(mm)
    @test r == 98
    # we no longer offer ordering guarantees besides preserving
    # relative order of linearly independent columns
    # and trying to keep the first column in the first position
    unpivoted = pivot[begin:r]
    @test unpivoted == sort(unpivoted)
end

@testset "zero columns in X" begin
    X = Matrix{Float64}(undef, 100, 0)
    r, pivot = statsrank(X)
    @test iszero(r)
    @test pivot == collect(axes(X, 2))
end
