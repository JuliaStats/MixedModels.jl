using Base.Test, RData, MixedModels

if !isdefined(:dat) || !isa(dat, Dict{Symbol, Any})
    dat = convert(Dict{Symbol,Any}, load(joinpath(dirname(@__FILE__), "dat.rda")))
end

@testset "scalarRe" begin
    dyestuff = dat[:Dyestuff]
    pastes = dat[:Pastes]
    sf = FactorReTerm(dyestuff[:G])
    sf1 = FactorReTerm(pastes[:G])
    sf2 = FactorReTerm(pastes[:H])
    Yield = Array(dyestuff[:Y])

    @testset "size" begin
        @test size(sf) == (30, 6)
        @test size(sf,1) == 30
        @test size(sf,2) == 6
        @test size(sf,3) == 1
        @test size(sf1) == (60, 30)
        @test size(sf2) == (60, 10)
    end

    @testset "utilities" begin
        @test MixedModels.levs(sf) == string.('A':'F')
        @test MixedModels.nlevs(sf) == 6
        @test MixedModels.vsize(sf) == 1
        @test MixedModels.nrandomeff(sf) == 6
        @test eltype(sf) == Float64
        @test sparse(sf) == sparse(Int32[1:30;], convert(Vector{Int32},sf.f.refs), ones(30))
        fsf = full(sf)
        @test size(fsf) == (30, 6)
        @test countnz(fsf) == 30
        @test sort!(unique(fsf)) == [0.0, 1.0]
        @test cond(sf) == 1.0
        @test MixedModels.nθ(sf) == 1
        @test getθ(sf) == ones(1)
        @test MixedModels.getθ!(Vector{Float64}(1), sf) == ones(1)
        @test lowerbd(sf) == zeros(1)
        @test eltype(sf) == Float64
        @test getθ(setθ!(sf, [0.5])) == [0.5]
        @test_throws DimensionMismatch MixedModels.getθ!(Float64[], sf)
        @test_throws DimensionMismatch setθ!(sf, ones(2))
    end

    @testset "products" begin
        dd = fill(5.0, (6, 1))
        @test sf'MatrixTerm(ones(30)) == dd
        @test MatrixTerm(ones(30))'sf == dd'
        @test Ac_mul_B!(Array{Float64}((size(sf1, 2), size(sf2, 2))), sf1, sf2) == Array(sf1'sf2)

        crp = sf'sf
        @test isa(crp, Diagonal{Float64})
        crp1 = copy(crp)
        @test crp1 == crp
        @test crp[2,6] == 0
        @test crp[6,6] == 5
        @test size(crp) == (6,6)
        @test crp.diag == fill(5.,6)
        rhs = sf'MatrixTerm(Yield)
        @test rhs == reshape([7525.0,7640.0,7820.0,7490.0,8000.0,7350.0], (6, 1))
        @test A_ldiv_B!(crp, copy(rhs)) == reshape([1505.,1528.,1564.,1498.,1600.,1470.], (6, 1))

        @test isa(sf1'sf1, Diagonal{Float64})
        @test isa(sf2'sf2, Diagonal{Float64})
        @test isa(sf2'sf1,SparseMatrixCSC{Float64})

        @test MixedModels.Λc_mul_B!(sf, ones(6)) == fill(0.5, 6)
        @test MixedModels.Λ_mul_B!(Vector{Float64}(6), sf, ones(6)) == fill(0.5, 6)
        @test MixedModels.A_mul_Λ!(ones(6, 6), sf) == fill(0.5, (6, 6))
    end

    @testset "reweight!" begin
        wts = rand(MersenneTwister(1234321), size(sf, 1))
        @test vec(MixedModels.reweight!(sf, wts).wtz) == wts
    end
end

@testset "vectorRe" begin
    slp = dat[:sleepstudy]
    corr = FactorReTerm(slp[:G], hcat(ones(size(slp, 1)), Array(slp[:U]))',
        :G, ["(Intercept)", "U"], [2])
    nocorr = FactorReTerm(slp[:G], hcat(ones(size(slp, 1)), Array(slp[:U]))',
            :G, ["(Intercept)", "U"], [1, 1])
    Reaction = Array(slp[:Y])

    @testset "sizes" begin
        @test size(corr) == (180,36)
        @test size(nocorr) == (180,36)
    end

    @testset "utilities" begin
        @test MixedModels.levs(corr) == DataArrays.levels(slp[:G])
        @test MixedModels.nlevs(corr) == 18
        @test MixedModels.vsize(corr) == 2
        @test MixedModels.nrandomeff(corr) == 36
        @test eltype(corr) == Float64
        @test nnz(sparse(corr)) == 360
        @test cond(corr) == 1.0
        @test MixedModels.nθ(corr) == 3
        @test MixedModels.nθ(nocorr) == 2
        @test getθ(corr) == [1.0, 0.0, 1.0]
        @test getθ(nocorr) == ones(2)
        @test MixedModels.getθ!(Vector{Float64}(2), nocorr) == ones(2)
        @test lowerbd(nocorr) == zeros(2)
        @test lowerbd(corr) == [0.0, -Inf, 0.0]
        @test getθ(setθ!(corr, fill(0.5, 3))) == [0.5, 0.5, 0.5]
        @test_throws DimensionMismatch MixedModels.getθ!(Vector{Float64}(2), corr)
        @test_throws DimensionMismatch setθ!(corr, ones(2))
    end

    @test vec(corr'MatrixTerm(ones(size(corr, 1)))) == repeat([10.0, 45.0], outer = 18)

    vrp = corr'corr
    @test isa(vrp, MixedModels.Diagonal{Matrix{Float64}})
    @test eltype(vrp) == Matrix{Float64}
    @test size(vrp) == (18, 18)

    @testset "reweight!" begin
        wts = rand(MersenneTwister(1234321), size(corr, 1))
        @test MixedModels.reweight!(corr, wts).wtz[1, :] == wts
        @test corr.z[1, :] == ones(size(corr, 1))
    end

end
