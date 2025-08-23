module MixedModelsPRIMAExt

using MixedModels
using MixedModels: Statistics
using MixedModels.ProgressMeter: ProgressMeter, ProgressUnknown
using MixedModels: objective!, _objective!, rectify!
using LinearAlgebra: PosDefException
using PRIMA: PRIMA

function __init__()
    push!(MixedModels.OPTIMIZATION_BACKENDS, :prima)
    return nothing
end

const PRIMABackend = Val{:prima}

function _optimizer(o::Val{O}, f, x0::Vector, args...; kwargs...) where {O}
    x0 = copy(x0)
    info = _optimizer!(o, f, x0, args...; kwargs...)
    return x0, info
end

_optimizer!(::Val{:bobyqa}, args...; kwargs...) = PRIMA.bobyqa!(args...; kwargs...)
_optimizer!(::Val{:cobyla}, args...; kwargs...) = PRIMA.cobyla!(args...; kwargs...)
_optimizer!(::Val{:lincoa}, args...; kwargs...) = PRIMA.lincoa!(args...; kwargs...)
_optimizer!(::Val{:newuoa}, args...; kwargs...) = PRIMA.newuoa!(args...; kwargs...)

function MixedModels.optimize!(m::LinearMixedModel, ::PRIMABackend;
    progress::Bool=true, kwargs...)
    optsum = m.optsum
    prog = ProgressUnknown(; desc="Minimizing", showspeed=true)
    empty!(optsum.fitlog)

    function obj(x)
        val = if x == optsum.initial
            # fast path since we've already evaluated the initial value
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
        push!(optsum.fitlog, (; θ=copy(x), objective=val))
        return val
    end

    maxfun = optsum.maxfeval > 0 ? optsum.maxfeval : 500 * length(optsum.initial)
    info = _optimizer!(Val(optsum.optimizer), obj, optsum.final;
        maxfun,
        optsum.rhoend, optsum.rhobeg)
    optsum.feval = info.nf
    optsum.fmin = info.fx
    optsum.returnvalue = Symbol(info.status)
    _check_prima_return(info)
    return optsum.final, optsum.fmin
end

function MixedModels.optimize!(m::GeneralizedLinearMixedModel, ::PRIMABackend;
    progress::Bool=true,
    fast::Bool=false, verbose::Bool=false, nAGQ=1,
    kwargs...)
    optsum = m.optsum
    prog = ProgressUnknown(; desc="Minimizing", showspeed=true)
    empty!(optsum.fitlog)

    function obj(x)
        val = try
            _objective!(m, x, Val(fast); verbose, nAGQ)
        catch ex
            # this allows us to recover from models where e.g. the link isn't
            # as constraining as it should be
            ex isa Union{PosDefException,DomainError} || rethrow()
            x == optsum.initial && rethrow()
            m.optsum.finitial
        end
        push!(optsum.fitlog, (; θ=copy(x), objective=val))
        verbose && println(round(val; digits=5), " ", x)
        progress && ProgressMeter.next!(prog; showvalues=[(:objective, val)])
        return val
    end

    optsum.finitial = _objective!(m, optsum.initial, Val(fast); verbose, nAGQ)
    maxfun = optsum.maxfeval > 0 ? optsum.maxfeval : 500 * length(optsum.initial)
    scale = if fast
        nothing
    else
        # scale by the standard deviation of the columns of the fixef model matrix
        # when including the fixef in the nonlinear opt
        sc = [map(std, eachcol(modelmatrix(m))); fill(1, length(m.θ))]
        for (i, x) in enumerate(sc)
            # for nearly constant things, e.g. intercept, we don't want to scale to zero...
            # also, since we're scaling the _parameters_ and not the data,
            # we need to invert the scale
            sc[i] = ifelse(iszero(x), one(x), inv(x))
        end
        sc
    end
    info = _optimizer!(Val(optsum.optimizer), obj, optsum.final;
        maxfun,
        optsum.rhoend, optsum.rhobeg,
        scale)
    ProgressMeter.finish!(prog)

    optsum.feval = info.nf
    optsum.fmin = info.fx
    optsum.returnvalue = Symbol(info.status)
    _check_prima_return(info)

    return optsum.final, optsum.fmin
end

function _check_prima_return(info::PRIMA.Info)
    if !PRIMA.issuccess(info)
        @warn "PRIMA optimization failure: $(info.status)\n$(PRIMA.reason(info))"
    end

    return nothing
end

MixedModels.opt_params(::PRIMABackend) = [:rhobeg, :rhoend, :maxfeval]
MixedModels.optimizers(::PRIMABackend) = [:bobyqa, :cobyla, :lincoa, :newuoa]

function MixedModels.profilevc(obj, optsum::OptSummary, ::PRIMABackend; kwargs...)
    maxfun = optsum.maxfeval > 0 ? optsum.maxfeval : 500 * length(optsum.initial)
    xmin, info = _optimizer(Val(optsum.optimizer), obj,
        copyto!(optsum.final, optsum.initial);
        maxfun,
        optsum.rhoend, optsum.rhobeg,
        scale=nothing) # will need to scale for GLMM
    _check_prima_return(info)
    fmin = info.fx
    return fmin, xmin
end

function MixedModels.profileobj!(obj,
    m::LinearMixedModel{T}, θ::AbstractVector{T}, osj::OptSummary, ::PRIMABackend;
    kwargs...) where {T}
    maxfun = osj.maxfeval > 0 ? osj.maxfeval : 500 * length(osj.initial)
    xmin = copyto!(osj.final, osj.initial)
    info = _optimizer!(Val(osj.optimizer), obj, xmin;
        maxfun,
        osj.rhoend, osj.rhobeg,
        scale=nothing) # will need to scale for GLMM
    fmin = info.fx
    _check_prima_return(info)
    rectify!(m)
    return fmin
end

end # module
