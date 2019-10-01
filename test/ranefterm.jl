using MixedModels, StatsModels

@testset "random effects term syntax" begin

    dat = (y = rand(18),
           g = string.(repeat('a':'f', inner=3)),
           f = string.(repeat('A':'C', outer=6)))
    
    @testset "fulldummy" begin
        @test_throws ArgumentError fulldummy(1)

        f = @formula(y ~ 1 + fulldummy(f))
        f1 = apply_schema(f, schema(dat))
        @test typeof(f1.rhs.terms[end]) <: FunctionTerm{typeof(fulldummy)}
        @test_throws ArgumentError modelcols(f1, dat)

        f2 = apply_schema(f, schema(dat), MixedModel)
        @test typeof(f2.rhs.terms[end]) <: CategoricalTerm{<:StatsModels.FullDummyCoding}
        @test modelcols(f2.rhs, dat)[1:3, :] == [1 1 0 0
                                                 1 0 1 0
                                                 1 0 0 1]

        # implict intercept
        ff = apply_schema(@formula(y ~ 1 + (f | g)), schema(dat), MixedModel)
        rem = modelcols(ff.rhs[end], dat) 
        size(rem) == (18, 18)
        @test rem[1:3, 1:4] == [1 0 0 0
                                1 1 0 0
                                1 0 1 0]

        # explicit intercept
        ff = apply_schema(@formula(y ~ 1 + (1+f | g)), schema(dat), MixedModel)
        rem = modelcols(ff.rhs[end], dat) 
        size(rem) == (18, 18)
        @test rem[1:3, 1:4] == [1 0 0 0
                                1 1 0 0
                                1 0 1 0]

        # explicit intercept + full dummy
        ff = apply_schema(@formula(y ~ 1 + (1+fulldummy(f) | g)), schema(dat), MixedModel)
        rem = modelcols(ff.rhs[end], dat)
        size(rem) == (18, 24)
        @test rem[1:3, 1:4] == [1 1 0 0
                                1 0 1 0
                                1 0 0 1]

        # explicit dropped intercept (implicit full dummy)
        ff = apply_schema(@formula(y ~ 1 + (0+f | g)), schema(dat), MixedModel)
        rem = modelcols(ff.rhs[end], dat)
        size(rem) == (18, 18)
        @test rem[1:3, 1:4] == [1 0 0 0
                                0 1 0 0
                                0 0 1 0]
        
    end
end
