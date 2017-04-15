using MixedModels, Base.Test

@testset "uniformscl" begin
    rng = MersenneTwister(1234321)
    const v = rand(rng, 8)
    const vc = copy(v)
    const M = rand(rng, (9, 9))
    const Mc = copy(M)
    const D = Diagonal(randn(rng, 9))
    const Dc = copy(D)
    const spm = sprand(rng, Float64, 10, 22, 0.2)
    const spmc = copy(spm)
    J = UniformScaling(1.0)
    @test A_mul_B!(similar(v), J, v) == v
    @test A_mul_B!(similar(M), J, M) == M
    @test A_mul_B!(similar(M), M, J) == M
    @test A_mul_B!(J, M) == Mc
    @test A_mul_B!(J, v) == vc
    @test A_mul_B!(M, J) == Mc
    @test A_mul_B!(v, J) == vc
    @test A_mul_B!(D, J) == Dc
    @test Ac_mul_B!(J, D) == Dc
    @test Ac_mul_B!(J, v) == vc
    @test Ac_mul_B!(J, M) == Mc
end

@testset "maskedltri" begin
    rng = MersenneTwister(1234321);
    const v = rand(rng, 8);
    const vc = copy(v);
    const M = rand(rng, (9, 9));
    const Mc = copy(M)
    const D = Diagonal(randn(rng, 9));
    const Dc = copy(D)
    const spm = sprand(rng, Float64, 10, 22, 0.2);
    const spmc = copy(spm)
    for m in (MixedModels.MaskedLowerTri(LowerTriangular(eye(1)), [1]),
        MixedModels.MaskedLowerTri([1], Float64), UniformScaling(1.0))
        @test getθ(m) == ones(1)
        @test MixedModels.getθ!(zeros(1), m) == ones(1)
        @test_throws DimensionMismatch MixedModels.getθ!(zeros(2), m)
        @test lowerbd(m) == zeros(1)
        @test cond(m) == 1
        @test MixedModels.nθ(m) == 1
        @test A_mul_B!(m, v) == vc
        @test A_mul_B!(M, m) == Mc
        if !isa(m, UniformScaling)
            @test setθ!(m,[0.5]) == MixedModels.MaskedLowerTri(LowerTriangular(fill(0.5,(1,1))), [1])
        end
    end

    for m in (MixedModels.MaskedLowerTri(LowerTriangular(eye(3)), [1,2,3,5,6,9]),
        MixedModels.MaskedLowerTri([3], Float64))
        @test getθ(m) == [1.,0.,0.,1.,0.,1.]
        @test lowerbd(m) == [0., -Inf, -Inf, 0., -Inf, 0.]
        setθ!(m, [1.:6;])
        @test m.m == reshape([1.,2,3,0,4,5,0,0,6], (3,3))
    end

    for m in (MixedModels.MaskedLowerTri(LowerTriangular(eye(3)), [1,5,9]),
        MixedModels.MaskedLowerTri([1,1,1], Float64))
        @test getθ(m) == ones(3)
        @test lowerbd(m) == zeros(3)
        setθ!(m, [1.:3;])
        @test m.m == reshape([1.,0,0,0,2,0,0,0,3], (3,3))
    end
end

@testset "identity" begin
    J = MixedModels.Identity{Float64}()
    @test cond(J) == 1.0
    @test MixedModels.getθ!(Float64[], J) == ones(0)
    @test lowerbd(J) == zeros(0)
    @test getθ(J) == ones(0)

    v = rand(3)
    vc = copy(v)
    M = rand(4,4)
    Mc = copy(M)
    @test A_mul_B!(similar(M), M, J) == M
    @test A_mul_B!(J, v) == vc
    @test A_mul_B!(J, M) == Mc
    @test A_mul_B!(J, v) == v
end

@testset "lambdavec" begin
    v = MixedModels.LambdaTypes{Float64}[]
    push!(v, MixedModels.MaskedLowerTri([3], Float64))
    push!(v, UniformScaling(0.5))
    push!(v, MixedModels.Identity{Float64}())
    @test MixedModels.nθ(v) == 7
    @test lowerbd(v) == [0., -Inf, -Inf, 0., -Inf, 0., 0.]
end
