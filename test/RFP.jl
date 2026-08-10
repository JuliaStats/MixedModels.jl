using LinearAlgebra
using MixedModels
using Test
using SparseArrays
using RectangularFullPacked

using MixedModels: rankUpdate!

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
