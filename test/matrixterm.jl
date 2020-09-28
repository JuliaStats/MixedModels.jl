using LinearAlgebra, MixedModels, StableRNGs, Test, SparseArrays

include("modelcache.jl")

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
    wts = rand(StableRNG(123454321), 30)
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

@testset "fematSparse" begin

    @testset "sparse and dense yield same fit" begin
        # deepcopy because we're going to modify
        m = deepcopy(last(models(:insteval)))
        # this is kinda sparse: 
        # julia> mean(first(m.feterms).x)
        # 0.10040140325753434
        
        fe = first(m.feterms)
        X =  MixedModels.FeMat(SparseMatrixCSC(fe.x), fe.cnames)
        @test typeof(X.x) <: SparseMatrixCSC
        @test typeof(X.wtx) <: SparseMatrixCSC
        @test X.rank == 28
        @test X.cnames == fe.cnames

        dense_θ = copy(m.θ)
        dense_feval = m.optsum.feval
        m.feterms[1] = X    
        refit!(m)

        @test dense_θ ≈ m.θ
    end

    @testset "rank defiency in sparse FeMat" begin
        trm = MixedModels.FeMat(SparseMatrixCSC(hcat(ones(30), 
                                                     repeat(0:9, outer = 3), 
                                                     2repeat(0:9, outer = 3))), 
                                ["(Intercept)", "U", "V"])
        piv = trm.piv
        ipiv = invperm(piv)
        @test_broken rank(trm) == 2
        @test size(trm) == (30, 3)
        @test length(trm) == 90
        @test size(trm') == (3, 30)
        @test eltype(trm) == Float64
        @test trm.x === trm.wtx
        prd = trm'trm
        @test typeof(prd) == typeof(prd)
        @test prd == [30.0 135.0 270.0; 135.0 855.0 1710.0; 270.0 1710.0 3420][piv, piv]
        wts = rand(StableRNG(123454321), 30)
        MixedModels.reweight!(trm, wts)
        @test mul!(prd, trm', trm)[ipiv[1], ipiv[1]] ≈ sum(abs2, wts)
    end
    
end