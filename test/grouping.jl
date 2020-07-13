using Test
using StatsModels

@testset "Grouping pseudo-contrasts" begin
    d = (y = rand(2_000_000), grp=string.([1:1_000_000; 1:1_000_000]))
    @test_throws OutOfMemoryError schema(d)
    sch = schema(d, Dict(:grp => Grouping()))
    t = sch[term(:grp)]
    @test t isa CategoricalTerm{Grouping}
    @test size(t.contrasts.matrix) == (0,0)
    @test length(t.contrasts.levels) == 1_000_000

    levs = sort(string.(1:1_000_000))

    @test all(t.contrasts.invindex[lev] == i for (i,lev) in enumerate(levs))
    @test all(t.contrasts.levels[i] == lev for (i,lev) in enumerate(levs))
end
