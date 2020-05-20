using LinearAlgebra, MixedModels, StatsModels, Test

xtx(X) = X'X  # creat the symmetric matrix X'X from X

const simdat = (
    G = repeat('A':'T', inner=10),
    H = repeat('a':'e', inner=2, outer=20),
    U = repeat(0.:9, outer=20),
    V = repeat(-4.5:4.5, outer=20),
    Y = 1:200,
    Z = 1:200
)

@testset "fullranknumeric" begin
    XtX = xtx(modelmatrix(@formula(Y ~ 1 + U), simdat))
    ch = statscholesky(Symmetric(XtX, :U))
    @test ch.rank == 2
    @test ch.piv == 1:2
    @test iszero(ch.info)
    @test isapprox(xtx(ch.U), XtX[ch.piv, ch.piv])
end

@testset "fullrankcategorical" begin
    XtX = xtx(modelmatrix(@formula(Y ~ 1 + G*H), simdat))
    ch = statscholesky(Symmetric(XtX, :U))
    @test ch.rank == 100
    @test ch.piv == 1:100
    @test iszero(ch.info)
    @test isapprox(xtx(ch.U), XtX)
end

@testset "dependentcolumn" begin
    XtX = xtx(modelmatrix(@formula(Y ~ 1 + U + V + Z), simdat))
    ch = statscholesky(Symmetric(XtX, :U))
    perm = [1,2,4,3]
    @test ch.rank == 3
    @test ch.piv == perm
    @test isapprox(xtx(ch.U), XtX[perm, perm])
end

@testset "missingcells" begin
    XtX = xtx(modelmatrix(@formula(Y ~ 1 + G*H), simdat)[5:end,:])
    ch = statscholesky(Symmetric(XtX, :U))
    perm = [1:42; 44:100; 43]
    @test ch.rank == 98
    @test ch.piv == perm
    @test isapprox(xtx(ch.U), XtX[perm, perm], atol=0.00001)
end
