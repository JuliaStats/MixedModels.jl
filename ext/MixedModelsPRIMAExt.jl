module MixedModelsPRIMAExt

using MixedModels: MixedModels, LinearMixedModel, objective!
using MixedModels: ProgressMeter, ProgressUnknown
using PRIMA: PRIMA

function MixedModels.prfit!(m::LinearMixedModel;
                            kwargs...)

    unfit!(m)
    m.optsum.optimizer = :bobyqa
    m.optsum.backend = :prima

    return fit!(m; kwargs...)
end

prima_optimizer!(::Val{:bobyqa}, args...; kwargs...) = PRIMA.bobyqa!(args...; kwargs...)
prima_optimizer!(::Val{:cobyla}, args...; kwargs...) = PRIMA.cobyla(args...; kwargs...)
prima_optimizer!(::Val{:lincoa}, args...; kwargs...) = PRIMA.lincoa(args...; kwargs...)

push!(MixedModels.OPTIMIZATION_BACKENDS, :prima)

const PRIMABackend = Val{:prima}

function MixedModels.optimize!(m::LinearMixedModel, ::PRIMABackend; progress::Bool=true, thin::Int=tyepmax(Int))
    optsum = m.optsum
    prog = ProgressUnknown(; desc="Minimizing", showspeed=true)
    # start from zero for the initial call to obj before optimization
    iter = 0
    fitlog = optsum.fitlog

    function obj(x)
        isempty(g) || throw(ArgumentError("g should be empty for this objective"))
        iter += 1
        val = if isone(iter) && x == optsum.initial
            optsum.finitial
        else
            try
                objective!(m, x)
            catch ex
                # This can happen when the optimizer drifts into an area where
                # there isn't enough shrinkage. Why finitial? Generally, it will
                # be the (near) worst case scenario value, so the optimizer won't
                # view it as an optimum. Using Inf messes up the quadratic
                # approximation in BOBYQA.
                ex isa PosDefException || rethrow()
                optsum.finitial
            end
        end
        progress && ProgressMeter.next!(prog; showvalues=[(:objective, val)])
        !isone(iter) && iszero(rem(iter, thin)) && push!(fitlog, (copy(x), val))
        return val
    end

    try
        # use explicit evaluation w/o calling opt to avoid confusing iteration count
        optsum.finitial = objective!(m, optsum.initial)
    catch ex
        ex isa PosDefException || rethrow()
        # give it one more try with a massive change in scaling
        @info "Initial objective evaluation failed, rescaling initial guess and trying again."
        @warn """Failure of the initial evaluation is often indicative of a model specification
                that is not well supported by the data and/or a poorly scaled model.
            """
        optsum.initial ./=
            (isempty(m.sqrtwts) ? 1.0 : maximum(m.sqrtwts)^2) *
            maximum(response(m))
        optsum.finitial = objective!(m, optsum.initial)
    end
    empty!(fitlog)
    push!(fitlog, (copy(optsum.initial), optsum.finitial))
    info = prima_optimizer!(Val(optsum.optimizer), obj, optsum.final; xl=m.optsum.lowerbd)
    ProgressMeter.finish!(prog)
    optsum.feval = info.nf
    optsum.fmin = info.fx
    optsum.returnvalue = Symbol(info.status)

    return optsum.final, optsum.fmin
end


function _check_prima_return(info::PRIMA.Info)
    if !PRIMA.issuccess(info)
        @warn "PRIMA optimization failure: $(ret)\n$(PRIMA.reason(info))"
    end

    return nothing
end

MixedModels.opt_params(::PRIMABackend) = (:rhobeg, :rhoend)

end # module
