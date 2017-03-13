@testset "vectorRe" begin
    vf = remat(:(1 + U | G), sleepstudy)
    Reaction = Array(sleepstudy[:Y])

    sf = remat(:(1 | G), sleepstudy)

    @test size(vf) == (180,36)
    @test (vf'ones(size(vf,1)))[1:4] == [10.,45,10,45]

    vrp = vf'vf
    @test isa(vrp, MixedModels.Diagonal{Matrix{Float64}})
    @test eltype(vrp) == Matrix{Float64}
    @test size(vrp) == (18, 18)
#    @test Ac_mul_B!(Array(Float64, (36, 36)), vf, vf) == full(vrp)

    D = Diagonal(Array(1.:180))
    scaled = D * vf
    @test scaled == A_mul_B!(D, vf)

    pr = vf'sf
    @test size(pr) == (36, 18)
    @test isa(pr, SparseMatrixCSC)
    @test size(sf'vf) == (18, 36)

end
