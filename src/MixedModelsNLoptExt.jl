module MixedModelsNLoptExt # not actually an extension at the moment

using ..MixedModels
using ..MixedModels:
    objective!, _objective!, rectify!, ssqdenom,
    GradientWorkspace, objective_gradient!
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
    progress::Bool=true,
    kwargs...)
    optsum = m.optsum
    prog = ProgressUnknown(; desc="Minimizing", showspeed=true)
    empty!(optsum.fitlog)
    # gradient workspace, created only for the LD_* optimizers
    gradws = if !startswith(string(optsum.optimizer), "LD")
        nothing
    elseif optsum.gradient == :analytic
        GradientWorkspace(m)
    elseif optsum.gradient == :forwarddiff
        hasmethod(MixedModels.fd_gradient_workspace, Tuple{typeof(m)}) ||
            throw(
                ArgumentError(
                    "gradient=:forwarddiff requires that ForwardDiff.jl be loaded, e.g. `using ForwardDiff`"
                ),
            )
        MixedModels.fd_gradient_workspace(m)
    else
        throw(
            ArgumentError(
                "gradient must be :analytic or :forwarddiff, got $(optsum.gradient)"),
        )
    end
    # The gradient-based optimizers see the per-observation objective: the objective
    # and its gradient scale with the number of observations, so on that scale the
    # identity is a poor initial inverse-Hessian estimate and the first line-search
    # step of e.g. LBFGS overshoots wildly for large data sets.  fitlog, the progress
    # display, and the returned fmin remain on the usual deviance scale.
    scale = isnothing(gradws) ? 1.0 : Float64(ssqdenom(m))

    function obj(x, g)
        isnothing(gradws) && !isempty(g) &&
            throw(ArgumentError("g should be empty for this objective"))
        val = if isempty(g) && x == optsum.initial
            # fast path since we've already evaluated the initial value
            optsum.finitial
        else
            try
                if isempty(g)
                    objective!(m, x)
                else
                    val′ = _objective_gradient!(gradws, g, m, x)
                    g ./= scale
                    val′
                end
            catch ex
                # This can happen when the optimizer drifts into an area where
                # there isn't enough shrinkage. Why finitial? Generally, it will
                # be the (near) worst case scenario value, so the optimizer won't
                # view it as an optimum. Using Inf messes up the quadratic
                # approximation in BOBYQA. A zero gradient is the least harmful
                # signal we can hand a line search in that state.
                ex isa PosDefException || rethrow()
                isempty(g) || fill!(g, false)
                optsum.finitial
            end
        end
        progress && ProgressMeter.next!(prog; showvalues=[(:objective, val)])
        push!(optsum.fitlog, (; θ=copy(x), objective=val))
        return val / scale
    end

    # ftol_rel is invariant under the scaling; ftol_abs applies to the scaled
    # objective, i.e. it acts as a per-observation absolute tolerance
    opt = Opt(optsum)
    NLopt.min_objective!(opt, obj)
    fmin, xmin, ret = NLopt.optimize!(opt, copyto!(optsum.final, optsum.initial))
    fmin *= scale
    ProgressMeter.finish!(prog)
    optsum.feval = opt.numevals
    optsum.returnvalue = ret
    _check_nlopt_return(ret)
    return xmin, fmin
end

# dispatch between the analytic gradient and a gradient source provided by an
# extension (whose workspace type is not nameable here)
function _objective_gradient!(w::GradientWorkspace, g, m::LinearMixedModel, x)
    return objective_gradient!(w, g, m, x)
end
function _objective_gradient!(w, g, m::LinearMixedModel, x)
    return MixedModels.fd_objective_gradient!(w, g, m, x)
end

function MixedModels.optimize!(m::GeneralizedLinearMixedModel, ::NLoptBackend;
    progress::Bool=true,
    fast::Bool=false, verbose::Bool=false, nAGQ=1,
    kwargs...)
    optsum = m.optsum
    prog = ProgressUnknown(; desc="Minimizing", showspeed=true)
    empty!(optsum.fitlog)

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
        push!(optsum.fitlog, (; θ=copy(x), objective=val))
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

function NLopt.Opt(optsum::OptSummary, optimizer::Symbol=optsum.optimizer)
    n = length(optsum.initial)

    if optimizer == :LN_NEWUOA && isone(n) # :LN_NEWUOA doesn't allow n == 1
        optimizer = optsum.optimizer = :LN_BOBYQA
    end
    opt = NLopt.Opt(optimizer, n)
    NLopt.ftol_rel!(opt, optsum.ftol_rel) # relative criterion on objective
    NLopt.ftol_abs!(opt, optsum.ftol_abs) # absolute criterion on objective
    NLopt.xtol_rel!(opt, optsum.xtol_rel) # relative criterion on parameter values
    if length(optsum.xtol_abs) == n  # not true for fast=false optimization in GLMM
        NLopt.xtol_abs!(opt, optsum.xtol_abs) # absolute criterion on parameter values
    end
    NLopt.maxeval!(opt, optsum.maxfeval)
    NLopt.maxtime!(opt, optsum.maxtime)
    if isempty(optsum.initial_step)
        optsum.initial_step = NLopt.initial_step(opt, optsum.initial)
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
    return [:LN_NEWUOA, :LN_BOBYQA, :LN_COBYLA, :LN_NELDERMEAD, :LN_PRAXIS,
        :LD_LBFGS, :LD_MMA, :LD_SLSQP]
end

# the profiling objectives do not evaluate gradients, so profiling a model that was
# fitted with a gradient-based optimizer falls back to a derivative-free one
function _derivfree(optimizer::Symbol)
    return startswith(string(optimizer), "LD") ? :LN_BOBYQA : optimizer
end

function MixedModels.profilevc(obj, optsum::OptSummary, ::NLoptBackend; kwargs...)
    opt = NLopt.Opt(optsum, _derivfree(optsum.optimizer))
    NLopt.min_objective!(opt, obj)
    fmin, xmin, ret = NLopt.optimize!(opt, copyto!(optsum.final, optsum.initial))
    _check_nlopt_return(ret)

    return fmin, xmin
end

function MixedModels.profileobj!(obj,
    m::LinearMixedModel{T}, θ::AbstractVector{T}, osj::OptSummary, ::NLoptBackend;
    kwargs...) where {T}
    opt = NLopt.Opt(osj, _derivfree(osj.optimizer))
    NLopt.min_objective!(opt, obj)
    fmin, xmin, ret = NLopt.optimize(opt, copyto!(osj.final, osj.initial))
    _check_nlopt_return(ret)
    rectify!(m)
    return fmin
end

end # module
