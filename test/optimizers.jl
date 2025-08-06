using PRIMA, MixedModels, StatsModels, TypedTables

include("./modelcache.jl")

function compareopts(
    ff::StatsModels.FormulaTerm,
    dd;
    opts=@NamedTuple{bcknd::Symbol, opt::Symbol}[
        (:prima, :newuoa),
        (:prima, :bobyqa),
        (:prima, :cobyla),
        (:nlopt, :LN_BOBYQA),
        (:nlopt, :LN_NEWUOA),
        (:nlopt, :LN_COBYLA),
    ],
)
    res = @NamedTuple{bcknd::Symbol, optimizer::Symbol, neval::Int, obj::Float64}[]
    for opt in opts
        try
            opsum =
                fit(
                    MixedModel, ff, dd; progress=false, backend=opt.bcknd, optimizer=opt.opt
                ).optsum
            push!(res, (opt.bcknd, opt.opt, opsum.feval, opsum.fmin))
        catch
            return opt
        end
    end
    return Table(res)
end
