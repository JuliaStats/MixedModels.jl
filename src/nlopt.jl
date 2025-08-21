module MixedModelsNLoptExt # not actually an extension at the moment

using ..MixedModels
using ..MixedModels: objective!, _objective!
# are part of the package's dependencies and will not be part
# of the extension's dependencies
using ..MixedModels.ProgressMeter: ProgressMeter, ProgressUnknown

# stdlib
using LinearAlgebra: PosDefException
# will be a weakdep when this is moved to an extension
using NLopt: NLopt, Opt

function __init__()
    push!(MixedModels.OPTIMIZATION_BACKENDS, :nlopt)
    return nothing
end

const NLoptBackend = Val{:nlopt}

function MixedModels.optimize!(m::LinearMixedModel, ::NLoptBackend;
    progress::Bool=true, fitlog::Bool=false,
    kwargs...)
    optsum = m.optsum
    prog = ProgressUnknown(; desc="Minimizing", showspeed=true)
    fitlog && empty!(optsum.fitlog)

    function obj(x, g)
        isempty(g) || throw(ArgumentError("g should be empty for this objective"))
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
        fitlog && push!(optsum.fitlog, (copy(x), val))
        return val
    end

    opt = Opt(optsum)
    NLopt.min_objective!(opt, obj)
    fmin, xmin, ret = NLopt.optimize!(opt, copyto!(optsum.final, optsum.initial))
    ProgressMeter.finish!(prog)
    optsum.feval = opt.numevals
    optsum.returnvalue = ret
    _check_nlopt_return(ret)
    return xmin, fmin
end

function MixedModels.optimize!(m::GeneralizedLinearMixedModel, ::NLoptBackend;
    progress::Bool=true, fitlog::Bool=false,
    fast::Bool=false, verbose::Bool=false, nAGQ=1,
    kwargs...)
    optsum = m.optsum
    prog = ProgressUnknown(; desc="Minimizing", showspeed=true)
    fitlog && empty!(optsum.fitlog)

    function obj(x, g)
        isempty(g) || throw(ArgumentError("g should be empty for this objective"))
        val = try
            _objective!(m, x, Val(fast); verbose, nAGQ)
        catch ex
            # this allows us to recover from models where e.g. the link isn't
            # as constraining as it should be
            ex isa Union{PosDefException,DomainError} || rethrow()
            x == optsum.initial && rethrow()
            optsum.finitial
        end
        fitlog && push!(optsum.fitlog, (copy(x), val))
        verbose && println(round(val; digits=5), " ", x)
        progress && ProgressMeter.next!(prog; showvalues=[(:objective, val)])
        return val
    end

    opt = Opt(optsum)
    NLopt.min_objective!(opt, obj)
    optsum.finitial = _objective!(m, optsum.initial, Val(fast); verbose, nAGQ)
    fmin, xmin, ret = NLopt.optimize(opt, copyto!(optsum.final, optsum.initial))
    ProgressMeter.finish!(prog)

    optsum.feval = opt.numevals
    optsum.returnvalue = ret
    _check_nlopt_return(ret)

    return xmin, fmin
end

function NLopt.Opt(optsum::OptSummary)
    lb = optsum.lowerbd
    n = length(lb)

    if optsum.optimizer == :LN_NEWUOA && isone(n) # :LN_NEWUOA doesn't allow n == 1
        optsum.optimizer = :LN_BOBYQA
    end
    opt = NLopt.Opt(optsum.optimizer, n)
    NLopt.ftol_rel!(opt, optsum.ftol_rel) # relative criterion on objective
    NLopt.ftol_abs!(opt, optsum.ftol_abs) # absolute criterion on objective
    NLopt.xtol_rel!(opt, optsum.xtol_rel) # relative criterion on parameter values
    if length(optsum.xtol_abs) == n  # not true for fast=false optimization in GLMM
        NLopt.xtol_abs!(opt, optsum.xtol_abs) # absolute criterion on parameter values
    end
    #    NLopt.lower_bounds!(opt, lb)   # use unconstrained optimization even for :LN_BOBYQA
    NLopt.maxeval!(opt, optsum.maxfeval)
    NLopt.maxtime!(opt, optsum.maxtime)
    if isempty(optsum.initial_step)
        optsum.initial_step = NLopt.initial_step(opt, optsum.initial, similar(lb))
    else
        NLopt.initial_step!(opt, optsum.initial_step)
    end
    return opt
end

const _NLOPT_FAILURE_MODES = [
    :FAILURE,
    :INVALID_ARGS,
    :OUT_OF_MEMORY,
    :FORCED_STOP,
    :MAXEVAL_REACHED,
    :MAXTIME_REACHED,
]

function _check_nlopt_return(ret, failure_modes=_NLOPT_FAILURE_MODES)
    ret == :ROUNDOFF_LIMITED && @warn("NLopt was roundoff limited")
    if ret ∈ failure_modes
        @warn("NLopt optimization failure: $ret")
    end
end

function MixedModels.opt_params(::NLoptBackend)
    return [:ftol_rel, :ftol_abs, :xtol_rel, :xtol_abs, :initial_step, :maxfeval, :maxtime]
end

function MixedModels.optimizers(::NLoptBackend)
    return [:LN_NEWUOA, :LN_BOBYQA, :LN_COBYLA, :LN_NELDERMEAD, :LN_PRAXIS]
end

end # module
