push!(OPTIMIZATION_BACKENDS, :nlopt)

const NLoptBackend = Val{:nlopt}

function optimize!(m::LinearMixedModel, ::NLoptBackend; progress::Bool=true, thin::Int=tyepmax(Int))
    optsum = m.optsum
    opt = Opt(optsum)

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
    prog = ProgressUnknown(; desc="Minimizing", showspeed=true)
    # start from zero for the initial call to obj before optimization
    iter = 0
    fitlog = optsum.fitlog

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
    fmin, xmin, ret = NLopt.optimize!(opt, copyto!(optsum.final, optsum.initial))
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
