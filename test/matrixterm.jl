using LinearAlgebra, MixedModels, StableRNGs, Test, SparseArrays

include("modelcache.jl")

@testset "Xymat" begin
    trm = MixedModels.FeTerm(hcat(ones(30), repeat(0:9, outer = 3)), ["(Intercept)", "U"])
    piv = trm.piv
    ipiv = invperm(piv)
    mat = MixedModels.FeMat(trm, Float64.(collect(axes(trm.x, 1))))
    @test size(mat) == (30, 3)
    @test length(mat) == 90
    @test size(mat') == (3, 30)
    @test eltype(mat) == Float64
    @test mat.xy === mat.wtxy
    prd = mat'mat
    @test typeof(prd) == Matrix{Float64}
    @test prd[ipiv, ipiv] == [30.0 135.0; 135.0 855.0]
    wts = rand(StableRNG(123454321), 30)
    MixedModels.reweight!(mat, wts)
    @test mul!(prd, mat', mat)[ipiv[1], ipiv[1]] ≈ sum(abs2, wts)

    # empty fixed effects
    trm = MixedModels.FeTerm(ones(10,0), String[])
    #@test size(trm) == (10, 0)  # There no longer are size and length methods for FeTerm
    #@test length(trm) == 0
    #@test size(trm') == (0, 10)
    @test eltype(trm) == Float64
    @test trm.rank == 0
end

@testset "XymatSparse" begin

    @testset "sparse and dense yield same fit" begin
        # deepcopy because we're going to modify
        m = last(models(:insteval))
        # this is kinda sparse: 
        # julia> mean(first(m.feterm).x)
        # 0.10040140325753434
        
        fe = m.feterm
        X =  MixedModels.FeTerm(SparseMatrixCSC(fe.x), fe.cnames)
        @test typeof(X.x) <: SparseMatrixCSC
        @test X.rank == 28
        @test X.cnames == fe.cnames
        m1 = LinearMixedModel(collect(m.y), X, deepcopy(m.reterms), m.formula)
        # because of the way the initial values are calculated
        # m1.optsum.initial == m.optsum.final at this point
        copyto!(m1.optsum.initial, m.optsum.initial)
        fit!(m1; progress=false)
        @test isapprox(m1.θ, m.θ, rtol = 1.0e-5)
    end

    @testset "rank deficiency in sparse FeTerm" begin
        trm = MixedModels.FeTerm(SparseMatrixCSC(hcat(ones(30), 
                                                     repeat(0:9, outer = 3), 
                                                     2repeat(0:9, outer = 3))), 
                                ["(Intercept)", "U", "V"])
        # at present there is no attempt to evaluate the rank of a SparseMatrixCSC
        piv = trm.piv
        ipiv = invperm(piv)
        @test_broken rank(trm) == 2
        mat = MixedModels.FeMat(trm, Float64.(collect(axes(trm.x, 1))))
        @test size(mat) == (30, 4)
        @test length(mat) == 120
        @test size(mat') == (4, 30)
        @test eltype(mat) == Float64
        @test mat.xy === mat.wtxy
        prd = MixedModels.densify(mat'mat)
        @test typeof(prd) == typeof(prd)
        @test prd[ipiv, ipiv] == [30.0 135.0 270.0; 135.0 855.0 1710.0; 270.0 1710.0 3420.0]
        wts = rand(StableRNG(123454321), 30)
        MixedModels.reweight!(mat, wts)
        @test mul!(prd, mat', mat)[ipiv[1], ipiv[1]] ≈ sum(abs2, wts)
    end
    
end
