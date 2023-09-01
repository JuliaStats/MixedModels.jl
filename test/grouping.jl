using MixedModels
using StatsModels
using Test

using MixedModels: schematize
using StatsModels: ContrastsMatrix, FullDummyCoding

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
end

@testset "Auto application of Grouping()" begin

    d = (; y=rand(100),
        x=rand('A':'Z', 100),
        z=rand('A':'Z', 100),
        grp=rand('a':'z', 100))
    contrasts = Dict{Symbol, Any}()

    @testset "blocking variables are grouping" for f in [@formula(y ~ 1 + x + (1|grp)),
                                                         @formula(y ~ 1 + x + zerocorr(1|grp))]
        fsch = schematize(f, d, contrasts)
        fe = fsch.rhs[1]
        x = last(fe.terms)
        @test x.contrasts isa ContrastsMatrix{DummyCoding}
        re = fsch.rhs[2]
        grp = re.rhs
        @test grp.contrasts isa ContrastsMatrix{Grouping}
    end

    @testset "FE contrasts take priority" for f in [@formula(y ~ 1 + x + (1|x)),
                                                    @formula(y ~ 1 + x + zerocorr(1|x))]
        fsch = schematize(f, d, contrasts)
        fe = fsch.rhs[1]
        x = last(fe.terms)
        @test x.contrasts isa ContrastsMatrix{DummyCoding}
        re = fsch.rhs[2]
        grp = re.rhs
        @test grp.contrasts isa ContrastsMatrix{DummyCoding}

        fsch = schematize(@formula(y ~ 1 + x + (1|x)), d, Dict(:x => EffectsCoding()))
        fe = fsch.rhs[1]
        x = last(fe.terms)
        @test x.contrasts isa ContrastsMatrix{EffectsCoding}
        re = fsch.rhs[2]
        grp = re.rhs
        @test grp.contrasts isa ContrastsMatrix{EffectsCoding}
    end

    @testset "Nesting and interactions" for f in [@formula(y ~ 1 + x + (1|grp/z))]
        # XXX zerocorr(1|grp/z) doesn't work!
        fsch = schematize(f, d, contrasts)
        fe = fsch.rhs[1]
        x = last(fe.terms)
        @test x.contrasts isa ContrastsMatrix{DummyCoding}
        re = fsch.rhs[2:end]
        grp = re[1].rhs
        @test grp.contrasts isa ContrastsMatrix{Grouping}
        interaction = re[2].rhs
        # this is less than ideal but we need it for now to get the nesting logic to work
        @test interaction.terms[1].contrasts isa ContrastsMatrix{FullDummyCoding}
        # this is the desired behavior
        @test_broken interaction.terms[1].contrasts isa ContrastsMatrix{Grouping}
        @test interaction.terms[2].contrasts isa ContrastsMatrix{Grouping}
    end

    @testset "Interactions where one component is FE" for f in [@formula(y ~ 1 + x + (1|x&grp)),
                                                                @formula(y ~ 1 + x + zerocorr(1|x&grp))]
        # occurs in e.g. the contra models
        # @formula(use ~ 1+age+abs2(age)+urban+livch+(1|urban&dist)
        fsch = schematize(f, d, contrasts)
        fe = fsch.rhs[1]
        x = last(fe.terms)
        @test x.contrasts isa ContrastsMatrix{DummyCoding}
        re = fsch.rhs[2]
        x_re = re.rhs.terms[1]
        # this is less than ideal but it relates to the way interactions are computed in RE
        @test x_re.contrasts isa ContrastsMatrix{DummyCoding}
        # this is the desired behavior:
        # even if the contrast matrix has to be small enough to invert,
        # it's silly to store it and invert it again when we don't need it here
        @test_broken x_rex.contrasts isa ContrastsMatrix{Grouping}
        grp = re.rhs.terms[2]
        @test grp.contrasts isa ContrastsMatrix{Grouping}
    end

    @test_throws(ArgumentError("Same variable appears on both sides of |"),
                 schematize(@formula(y ~ 1 + (x|x)), d, contrasts))
    f1 = schematize(@formula(y ~ 1 + x + z), d, contrasts)
    f2 = apply_schema(@formula(y ~ 1 + x + z), schema(d, contrasts))
    # skip intercept term
    @test all(a.contrasts == b.contrasts for (a, b) in zip(f1.rhs.terms[2:end], f2.rhs.terms[2:end]))
end
