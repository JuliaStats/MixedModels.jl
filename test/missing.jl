using DataFrames
using MixedModels
using Test

# convert to DataFrame to modify it
slp = DataFrame(MixedModels.dataset(:sleepstudy))
allowmissing!(slp, :days)
slp[1,:days] = missing

# TODO: re-enable this test when better missing support has landed in StatsModels
# @testset "No impact from missing on schema" begin
#     f = @formula(Y ~ 1 + U + (1|G))
#     contrasts =  Dict{Symbol,Any}()
#     form = apply_schema(f, schema(f, dat[:sleepstudy], contrasts), LinearMixedModel)
#     form_missing = apply_schema(f, schema(f, slp, contrasts), LinearMixedModel)
#
#     @test form.lhs == form_missing.lhs
#     @test form.rhs == form_missing.rhs
# end

@testset "Missing Omit" begin
    @testset "Missing from unused variables" begin
        # missing from unused variables should have no impact
        m1 = fit(MixedModel, @formula(reaction ~ 1 + (1|subj)), MixedModels.dataset(:sleepstudy), progress=false)
        m1_missing = fit(MixedModel, @formula(reaction ~ 1 + (1|subj)), slp, progress=false)
        @test isapprox(m1.θ, m1_missing.θ, rtol=1.0e-12)
    end

    @testset "Missing from used variables" begin
        m1 = fit(MixedModel, @formula(reaction ~ 1 + days + (1|subj)), MixedModels.dataset(:sleepstudy), progress=false)
        m1_missing = fit(MixedModel, @formula(reaction ~ 1 + days + (1|subj)), slp, progress=false)
        @test nobs(m1) - nobs(m1_missing) == 1
    end
end
