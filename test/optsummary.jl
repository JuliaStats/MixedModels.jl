using MixedModels
using MixedModelsDatasets: dataset
using Test

include("modelcache.jl")

@testset "opt limits" begin
    @testset "maxfeval" begin
        fm1 = LinearMixedModel(first(fms[:sleepstudy]), dataset(:sleepstudy))
        fm1.optsum.maxfeval = 1
        @test_logs (:warn, "NLopt optimization failure: MAXEVAL_REACHED") refit!(
            fm1; progress=false
        )
        @test fm1.optsum.returnvalue == :MAXEVAL_REACHED
        @test fm1.optsum.feval == 1
    end

    @testset "maxtime" begin
        # we need a big model to guarantee that we hit the time limit,
        # no matter how small
        fm1 = LinearMixedModel(last(fms[:kb07]), dataset(:kb07))
        maxtime = 1e-6
        fm1.optsum.maxtime = maxtime
        @test_logs (:warn, "NLopt optimization failure: MAXTIME_REACHED") fit!(
            fm1; progress=false
        )
        @test fm1.optsum.returnvalue == :MAXTIME_REACHED
        @test fm1.optsum.maxtime == maxtime
    end

    @testset "early stop returns an evaluated point" begin
        # NEWUOA can return a point built from uninitialized memory when it stops
        # during its initialization phase. Fill freed heap memory with garbage
        # so that the work array NLopt allocates is likely to reuse it.
        function dirty_heap!(val=12345.0)
            ptrs = [Ptr{Float64}(Libc.malloc(32_000)) for _ in 1:64]
            for p in ptrs, i in 1:4000
                unsafe_store!(p, val, i)
            end
            foreach(Libc.free, ptrs)
            return nothing
        end
        for setlimit! in (os -> os.maxfeval = 1, os -> os.maxtime = 1e-6)
            fm1 = LinearMixedModel(last(fms[:kb07]), dataset(:kb07))
            @test fm1.optsum.optimizer == :LN_NEWUOA
            setlimit!(fm1.optsum)
            dirty_heap!()
            @test_logs (:warn, r"NLopt optimization failure") fit!(fm1; progress=false)
            @test any(r -> r.θ == fm1.optsum.final, fm1.optsum.fitlog)
            @test fm1.optsum.fmin ≈ minimum(r -> r.objective, fm1.optsum.fitlog)
        end
    end
end
