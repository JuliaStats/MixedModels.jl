using LinearAlgebra, StableRNGs, StatsModels, Test

import MixedModels: statsqr, statscholesky

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
    st = statsqr(mm)
    @test st.p == 1:2
end

@testset "fullrankcategorical" begin
    mm = modelmatrix(@formula(Y ~ 1 + G*H), simdat)
    st = statsqr(mm)
    @test rank(st) == 100
    @test st.p == 1:100
end

@testset "dependentcolumn" begin
    mm = modelmatrix(@formula(Y ~ 1 + U + V + Z), simdat)
    st = statsqr(mm)
    perm = [1,2,4,3]
    @test rank(st) == 3
    @test st.p == perm
end

@testset "qr missing cells" begin
    mm = modelmatrix(@formula(Y ~ 1 + G*H), simdat)[5:end,:]
    st = statsqr(mm)
    @test rank(st) == 98
    # we no longer offer ordering guarantees besides preserving
    # relative order of linearly independent columns
    # and trying to keep the first column in the first position
    unpivoted = st.p[begin:rank(st)]
    @test unpivoted == sort(unpivoted)
end
