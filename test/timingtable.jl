using Chairmarks, PRIMA, TypedTables
include(joinpath(@__DIR__, "modelcache.jl"))

@isdefined(timingtable) || function timingtable(
    dsname::Symbol=:insteval,
    optimizers::Vector{NTuple{2,Symbol}}=
    [
        (:nlopt, :LN_NEWUOA),
        (:nlopt, :LN_BOBYQA),
        (:prima, :newuoa),
        (:prima, :bobyqa),
    ],
    seconds::Integer = 1;
)
    rowtype = @NamedTuple{
        modnum::Int,
        bkend::Symbol,
        optmzr::Symbol,
        feval::Int,
        objtiv::Float64,
        time::Float64,
    }
    val = rowtype[]
    mods = models(dsname)

    for (j, m) in enumerate(mods)
        for (bk, opt) in optimizers
            bmk = @b refit!(m; progress=false, backend=bk, optimizer=opt) seconds=seconds
            push!(val, rowtype((j, bk, opt, m.optsum.feval, m.optsum.fmin, bmk.time)))
        end
    end
    return Table(val)
end
