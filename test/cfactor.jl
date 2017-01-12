@testset "cfactor" begin
    de3 = Diagonal(ones(3))
    de2 = Diagonal(ones(2))
    dr3 = Diagonal(randn(3))

    @test_throws DimensionMismatch MixedModels.downdate!(dr3, de2, de3)
    MixedModels.downdate!(dr3, de3, de3)
    sp10 = sprand(10,3,0.2)
    MixedModels.downdate!(rand(3,3),sp10,sp10)
    MixedModels.downdate!(rand(3,3),sp10)
    f1 = ScalarReMat(pool(repeat(1:20, outer = 6)), ones(120), :S3, ["(Intercept)"])
    f2 = ScalarReMat(pool(repeat(string.(['a':'j';]), inner = 2, outer = 6)), ones(120), :S2, ["(Intercept)"])
    f3 = ScalarReMat(pool(repeat(string.(['A':'E';]), inner = 4, outer = 6)), ones(120), :S1, ["(Intercept)"])
    A = f1'f2
    B = f1'f3
    C = f2'f3
    CC = copy(C)
    @test MixedModels.downdate!(C, A, B) ≈ -5I * CC
end
