@testset "vectorRe" begin
    Days = convert(Vector{Float64}, slp[:Days])
    vf = VectorReMat(slp[:Subject], hcat(ones(length(Days)),Days)', :Subject,
        ["(Intercept)","Days"])
    Reaction = Array(slp[:Reaction])

    @test size(vf) == (180,36)
    @test (vf'ones(size(vf,1)))[1:4] == [10.,45,10,45]
    vrp = vf'vf

    @test isa(vrp, MixedModels.HBlkDiag{Float64})
    @test eltype(vrp) == Float64
    @test size(vrp) == (36,36)
    @test Ac_mul_B!(Array(Float64, (36, 36)), vf, vf) == full(vrp)
    rhs1 = ones(36,2)
    x = similar(rhs1)
    b1 = copy(vrp.arr[:,:,1]) + I
    @test view(MixedModels.inflate!(vrp).arr,:,:,1) == b1
    cf = cholfact(b1)
    @test triu!(view(cholfact!(vrp).arr,:,:,1)) == cf[:U]

    D = Diagonal(Array(1.:180))
    scaled = D * vf
    @test scaled == A_mul_B!(D, vf)

end
