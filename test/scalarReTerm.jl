@testset "scalarRe" begin
    sf = ScalarReMat(ds[:Batch], ones(size(ds,1)), :Batch, String["(Intercept)"])
    sf1 = ScalarReMat(psts[:Sample], ones(size(psts,1)), :sample, String["(Intercept)"])
    sf2 = ScalarReMat(psts[:Batch], ones(size(psts, 1)), :batch, String["(Intercept)"])
    Yield = Array(ds[:Yield])

    @testset "size" begin
        @test size(sf) == (30, 6)
        @test size(sf,1) == 30
        @test size(sf,2) == 6
        @test size(sf,3) == 1
    end

    @testset "products" begin
    dd = fill(5., 6)
    @test sf'ones(30) == dd
    @test ones(30)'sf == dd'
    tt = A_mul_B!(1., sf, dd, 0., zeros(30))
    @test tt == A_mul_B!(sf, dd, zeros(30))
    @test Ac_mul_B!(Array(Float64, (size(sf1, 2), size(sf2, 2))), sf1, sf2) == Array(sf1'sf2)

    crp = sf'sf
    @test isa(crp, Diagonal{Float64})
    crp1 = copy(crp)
    @test crp1 == crp
    @test crp[2,6] == 0
    @test crp[6,6] == 5
    @test size(crp) == (6,6)
    @test crp.diag == fill(5.,6)
    rhs = sf'Yield
    @test rhs == [7525.0,7640.0,7820.0,7490.0,8000.0,7350.0]
    @test A_ldiv_B!(crp,copy(rhs)) == [1505.,1528.,1564.,1498.,1600.,1470.]

    D = Diagonal(ones(30))
    csf = D * sf
    @test sf == csf
    @test sf == LinAlg.A_mul_B!(csf, D, sf)

#    @test sf == copy!(csf, sf)

    L = MixedModels.LT(sf)
    setθ!(L, [0.5])

    @test isa(MixedModels.tscale!(L, crp), Diagonal)
    @test crp.diag == fill(2.5, 6)
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
