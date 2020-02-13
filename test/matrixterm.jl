using LinearAlgebra, MixedModels, Random, Test

@testset "femat" begin
    trm = MixedModels.FeMat(hcat(ones(30), repeat(0:9, outer = 3)), ["(Intercept)", "U"])
    piv = trm.piv
    ipiv = invperm(piv)
    @test size(trm) == (30, 2)
    @test length(trm) == 60
    @test size(trm') == (2, 30)
    @test eltype(trm) == Float64
    @test trm.x === trm.wtx
    prd = trm'trm
    @test typeof(prd) == Matrix{Float64}
    @test prd == [30.0 135.0; 135.0 855.0][piv, piv]
    wts = rand(MersenneTwister(123454321), 30)
    MixedModels.reweight!(trm, wts)
    @test mul!(prd, trm', trm)[ipiv[1], ipiv[1]] ≈ sum(abs2, wts)

    # empty fixed effects
    trm = MixedModels.FeMat(ones(10,0), String[])
    @test size(trm) == (10, 0)
    @test length(trm) == 0
    @test size(trm') == (0, 10)
    @test eltype(trm) == Float64
    @test trm.rank == 0
end
