using LinearAlgebra, MixedModels, Random, Test

@testset "vectorterm" begin
    trm = MatrixTerm(ones(10))
    @test size(trm) == (10, 1)
    @test trm'trm == 10.0 * ones(1, 1)
    @test trm.cnames == [""]
    @test trm.x === trm.wtx
    wts = rand(MersenneTwister(1234321), 10)
    MixedModels.reweight!(trm, wts)
    @test !(trm.x === trm.wtx)
    @test trm'trm ≈ sum(abs2, wts) * ones(1, 1)
    @test mul!(Vector{eltype(trm)}(undef, 10), trm, ones(1)) == ones(10)
end

@testset "matrixterm" begin
    trm = MatrixTerm(hcat(ones(30), repeat(0:9, outer = 3)), ["(Intercept)", "U"])
    piv = trm.piv
    ipiv = invperm(piv)
    @test size(trm) == (30, 2)
    @test trm.x === trm.wtx
    prd = trm'trm
    @test typeof(prd) == Matrix{Float64}
    @test prd == [30.0 135.0; 135.0 855.0][piv, piv]
    wts = rand(MersenneTwister(123454321), 30)
    MixedModels.reweight!(trm, wts)
    @test mul!(prd, trm', trm)[ipiv[1], ipiv[1]] ≈ sum(abs2, wts)
end
