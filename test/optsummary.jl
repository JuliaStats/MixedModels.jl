using MixedModels
using MixedModels: dataset
using Test

include("modelcache.jl")

@testset "opt limits" begin
    @testset "maxfeval" begin
        fm1 = LinearMixedModel(first(fms[:sleepstudy]), dataset(:sleepstudy))
        fm1.optsum.maxfeval = 1
        @test_logs (:warn, "NLopt optimization failure: MAXEVAL_REACHED") refit!(fm1; progress=false)
        @test fm1.optsum.returnvalue == :MAXEVAL_REACHED
        @test fm1.optsum.feval == 1
    end

    @testset "maxtime" begin
        # we need a big model to guarantee that we hit the time limit,
        # no matter how small
        fm1 = LinearMixedModel(last(fms[:kb07]), dataset(:kb07))
        maxtime = 1e-6
        fm1.optsum.maxtime = maxtime
        @test_logs (:warn, "NLopt optimization failure: MAXTIME_REACHED") fit!(fm1; progress=false)
        @test fm1.optsum.returnvalue == :MAXTIME_REACHED
        @test fm1.optsum.maxtime == maxtime
    end
end
