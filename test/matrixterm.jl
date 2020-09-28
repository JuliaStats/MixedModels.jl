using LinearAlgebra, MixedModels, StableRNGs, Test, SparseArrays

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

@testset "fematSparse" begin
    ## Generate a sparse design matrix
    nrowsX = 50 # n events
    ncolsX = 10 # n predictors
    onsets = 5 .+ cumsum(Int.(round.((rand(StableRNG(1),nrowsX).*100)))) # minimal distance + random distance
    ncolsBasis = 30 # expand each ncolsX by 30

    X = rand(StableRNG(2),nrowsX,ncolsX) # generate predictors

    # generate Diagonal Basis Expansion Matrices
    basis = spdiagm(1 => 0.7*ones(ncolsBasis-1), 0 => 0.3*ones(ncolsBasis-1))
    bases = repeat([basis],length(onsets))

    # Generate sparse indices... for rows
    rows =  copy.(rowvals.(bases))
    for r in 1:length(rows)
        rows[r] .+= onsets[r]-1
    end
    rows = vcat(rows...)
    rows = repeat(rows,ncolsX)

    # ... for cols
    cols = []
    for Xcol in 1:ncolsX
        for b in 1:length(bases)
            for c in 1:ncolsBasis
                push!(cols,repeat([c+(Xcol-1)*ncolsBasis],length(nzrange(bases[b],c))))
            end
        end
    end
    cols = vcat(cols...)

    # ... for values
    vals = []
    for Xcol in 1:ncolsX
        push!(vals,vcat(nonzeros.(bases).*X[:,Xcol]...))
    end
    vals = vcat(vals...)

    # Generate the matrix
    Xdc = sparse(rows,cols,vals)

    # Actual testing starts here
    trm = MixedModels.FeMat(Xdc,string.(collect(range('a',length=10))))

    @test typeof(trm.x) <: SparseMatrixCSC

end
