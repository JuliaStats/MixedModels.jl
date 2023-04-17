using MixedModels
using StatsModels
using Test

@testset "Grouping" begin
    g = Grouping()
    @test isnothing(g.levels)
end

@testset "Grouping pseudo-contrasts" begin
    d = (; y=rand(2_000_000),
         grp=string.([1:1_000_000; 1:1_000_000]),
         outer=rand('A':'z', 2_000_000))
    ## OOM seems to result in the process being killed on Mac so this messes up CI
    # @test_throws OutOfMemoryError schema(d)
    sch = schema(d, Dict(:grp => Grouping()))
    t = sch[term(:grp)]
    @test t isa CategoricalTerm{Grouping}
    @test size(t.contrasts.matrix) == (0,0)
    @test length(t.contrasts.levels) == 1_000_000
    @test_throws ErrorException StatsModels.modelcols(t, (a = 1.,))

    levs = sort(string.(1:1_000_000))

    @test all(t.contrasts.invindex[lev] == i for (i,lev) in enumerate(levs))
    @test all(t.contrasts.levels[i] == lev for (i,lev) in enumerate(levs))

    # without auto-grouping, this OOM on most reasonable hardware because it default dummy coding
    # would mean constructing 1M x 1M matrix
    # nesting still inserts a full dummy
    # outer(Grouping:58→58)
    # outer(StatsModels.FullDummyCoding:58→58) & grp(Grouping:1000000→1000000)
    for f in [@formula(y ~ 0 + (1 | grp)),
              @formula(y ~ 0 + (1 | outer / grp)),
              @formula(y ~ 0 + (1 | outer & grp))]
        @test MixedModels._schematize(f, d, Dict{Symbol, Any}()) isa FormulaTerm
    end
end
