using MixedModels, RData, Test

if !@isdefined(dat) || !isa(dat, Dict{Symbol, DataFrame})
    const dat = Dict(Symbol(k) => v for (k, v) in
        load(joinpath(dirname(pathof(MixedModels)), "..", "test", "dat.rda")))
end

# deepcopy because we're going to modify it
slp = deepcopy(dat[:sleepstudy])
slp[!,:U] = Array{Union{Missing, Float64},1}(slp[!,:U])
slp[1,:U] = missing

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
        m1 = fit(MixedModel, @formula(Y ~ 1 + (1|G)), dat[:sleepstudy])
        m1_missing = fit(MixedModel, @formula(Y ~ 1 + (1|G)), slp)
        @test isapprox(m1.θ, m1_missing.θ, rtol=1.0e-12)
    end

    @testset "Missing from used variables" begin
        m1 = fit(MixedModel, @formula(Y ~ 1 + U + (1|G)), dat[:sleepstudy])
        m1_missing = fit(MixedModel, @formula(Y ~ 1 + U + (1|G)), slp)
        @test nobs(m1) - nobs(m1_missing) == 1
    end
end
