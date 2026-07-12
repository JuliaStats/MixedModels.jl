using Chain, Chairmarks, PRIMA, DataFrames
include(joinpath(@__DIR__, "modelcache.jl"))

@isdefined(timingtable) || function timingtable(;
    dsname::Symbol=:insteval,
    optimizers::Vector{NTuple{2,Symbol}}=
    [
        (:nlopt, :LN_NEWUOA),
        (:nlopt, :LN_BOBYQA),
        (:prima, :newuoa),
        (:prima, :bobyqa),
    ],
    seconds::Integer = 1
)
    rowtype = @NamedTuple{
        modnum::Int8,
        ntheta::Int8,
        dof::Int8,
        bkend::Symbol,
        optimizer::Symbol,
        feval::Int,
        objective::Float64,
        time::Float64,
    }
    val = rowtype[]
    mods = models(dsname)

    for (j, m) in enumerate(mods)
        ntheta = length(m.parmap)
        for (bk, opt) in optimizers
            bmk = @b refit!(m; progress=false, backend=bk, optimizer=opt) seconds=seconds
            push!(val, rowtype((j, ntheta, dof(m), bk, opt, m.optsum.feval, m.optsum.fmin, bmk.time)))
        end
    end
    return @chain DataFrame(val) begin
        groupby(:modnum)
        combine(
            :ntheta,
            :dof,
            :bkend,
            :optimizer,
            :feval,
            :objective,
            :time,
            :objective => minimum => :min_obj,
        )
        transform!([:objective, :min_obj] => ((x, y) -> x - y) => :del_obj)
    end
end

