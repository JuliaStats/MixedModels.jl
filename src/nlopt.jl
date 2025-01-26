push!(OPTIMIZATION_BACKENDS, :nlopt)

const NLoptBackend = Val{:nlopt}

function optimize!(m::LinearMixedModel, ::NLoptBackend;
                   progress::Bool=true, thin::Int=tyepmax(Int),
                   kwargs...)
    optsum = m.optsum
    opt = Opt(optsum)

    prog = ProgressUnknown(; desc="Minimizing", showspeed=true)
    function obj(x, g)
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
    NLopt.min_objective!(opt, obj)
    # start from zero for the initial call to obj before optimization
    iter = 0
    fitlog = optsum.fitlog

    empty!(fitlog)
    push!(fitlog, (copy(optsum.initial), optsum.finitial))
    fmin, xmin, ret = NLopt.optimize!(opt, copyto!(optsum.final, optsum.initial))
    ProgressMeter.finish!(prog)
    optsum.feval = opt.numevals
    optsum.returnvalue = ret
    _check_nlopt_return(ret)
    return xmin, fmin
end

function optimize!(m::GeneralizedLinearMixedModel, ::NLoptBackend;
                   progress::Bool=true, thin::Int=tyepmax(Int),
                   fast::Bool=false, verbose::Bool=false, nAGQ=1,
                   kwargs...)
    optsum = m.optsum
    fitlog = optsum.fitlog
    prog = ProgressUnknown(; desc="Minimizing", showspeed=true)
    # start from zero for the initial call to obj before optimization
    iter = 0
    fitlog = optsum.fitlog
    function obj(x, g)
        isempty(g) || throw(ArgumentError("g should be empty for this objective"))
        val = try
            _objective!(m, x, Val(fast); verbose, nAGQ)
        catch ex
            # this allows us to recover from models where e.g. the link isn't
            # as constraining as it should be
            ex isa Union{PosDefException,DomainError} || rethrow()
            iter == 1 && rethrow()
            m.optsum.finitial
        end
        iszero(rem(iter, thin)) && push!(fitlog, (copy(x), val))
        verbose && println(round(val; digits=5), " ", x)
        progress && ProgressMeter.next!(prog; showvalues=[(:objective, val)])
        iter += 1
        return val
    end
    opt = Opt(optsum)
    NLopt.min_objective!(opt, obj)
    optsum.finitial = _objective!(m, optsum.initial, Val(fast); verbose, nAGQ)
    empty!(fitlog)
    push!(fitlog, (copy(optsum.initial), optsum.finitial))
    fmin, xmin, ret = NLopt.optimize(opt, copyto!(optsum.final, optsum.initial))
    ProgressMeter.finish!(prog)

    optsum.feval = opt.numevals
    optsum.returnvalue = ret
    _check_nlopt_return(ret)

    return xmin, fmin
end

function NLopt.Opt(optsum::OptSummary)
    lb = optsum.lowerbd

    opt = NLopt.Opt(optsum.optimizer, length(lb))
    NLopt.ftol_rel!(opt, optsum.ftol_rel) # relative criterion on objective
    NLopt.ftol_abs!(opt, optsum.ftol_abs) # absolute criterion on objective
    NLopt.xtol_rel!(opt, optsum.xtol_rel) # relative criterion on parameter values
    if length(optsum.xtol_abs) == length(lb)  # not true for fast=false optimization in GLMM
        NLopt.xtol_abs!(opt, optsum.xtol_abs) # absolute criterion on parameter values
    end
    NLopt.lower_bounds!(opt, lb)
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

opt_params(::NLoptBackend) = (:ftol_rel, :ftol_abs, :xtol_rel, :xtol_abs, :initial_step, :maxfeval, :maxtime)
