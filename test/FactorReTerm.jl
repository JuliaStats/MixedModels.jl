using Base.Test, RData, MixedModels

if !isdefined(:dat) || !isa(dat, Dict{Symbol, Any})
    dat = convert(Dict{Symbol,Any}, load(joinpath(dirname(@__FILE__), "dat.rda")))
end

@testset "scalarRe" begin
    dyestuff = dat[:Dyestuff]
    pastes = dat[:Pastes]
    sf = FactorReTerm(dyestuff[:G], ones(1, size(dyestuff,1)), :Batch, ["(Intercept)"], [1])
    sf1 = FactorReTerm(pastes[:G], ones(1, size(pastes,1)), :sample, ["(Intercept)"], [1])
    sf2 = FactorReTerm(pastes[:H], ones(1, size(pastes, 1)), :batch, ["(Intercept)"], [1])
    Yield = Array(dyestuff[:Y])

    @testset "size" begin
        @test size(sf) == (30, 6)
        @test size(sf,1) == 30
        @test size(sf,2) == 6
        @test size(sf,3) == 1
    end

    @testset "products" begin
        dd = fill(5.0, (6, 1))
        @test sf'MatrixTerm(ones(30)) == dd
        @test MatrixTerm(ones(30))'sf == dd'
        tt = A_mul_B!(1., sf, dd, 0., zeros(30))
        @test tt == A_mul_B!(sf, dd, zeros(30))
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

        D = Diagonal(ones(30))
        csf = D * sf
        @test sf == csf
        @test sf == LinAlg.A_mul_B!(csf, D, sf)

#    @test sf == copy!(csf, sf)

        L = MixedModels.LT(sf, Dict{Symbol,Any}())
#    @test copy!(crp1, crp) == crp

        @test size(sf1) == (60, 30)
        @test size(sf2) == (60, 10)
        crp11 = sf1'sf1
        pr21 = sf2'sf1
        crp22 = sf2'sf2

        @test isa(crp11,Diagonal{Float64})
        @test isa(crp22,Diagonal{Float64})
        @test isa(pr21,SparseMatrixCSC{Float64})
    end
end
