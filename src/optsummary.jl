"""
    OptSummary

Summary of an `NLopt` optimization

# Fields
* `initial`: a copy of the initial parameter values in the optimization
* `finitial`: the initial value of the objective
* `lowerbd`: lower bounds on the parameter values
* `ftol_rel`: as in NLopt
* `ftol_abs`: as in NLopt
* `xtol_rel`: as in NLopt
* `xtol_abs`: as in NLopt
* `initial_step`: as in NLopt
* `maxfeval`: as in NLopt (`maxeval`)
* `maxtime`: as in NLopt
* `final`: a copy of the final parameter values from the optimization
* `fmin`: the final value of the objective
* `feval`: the number of function evaluations
* `optimizer`: the name of the optimizer used, as a `Symbol`
* `returnvalue`: the return value, as a `Symbol`
* `nAGQ`: number of adaptive Gauss-Hermite quadrature points in deviance evaluation for GLMMs
* `REML`: use the REML criterion for LMM fits
* `sigma`: a priori value for the residual standard deviation for LMM
* `fitlog`: A vector of tuples of parameter and objectives values from steps in the optimization

The latter four fields are MixedModels functionality and not related directly to the `NLopt` package or algorithms.

!!! note
    The internal storage of the parameter values within `fitlog` may change in
    the future to use a different subtype of `AbstractVector` (e.g., `StaticArrays.SVector`)
    for each snapshot without being considered a breaking change.
"""
mutable struct OptSummary{T<:AbstractFloat}
    initial::Vector{T}
    lowerbd::Vector{T}
    finitial::T
    ftol_rel::T
    ftol_abs::T
    xtol_rel::T
    xtol_abs::Vector{T}
    initial_step::Vector{T}
    maxfeval::Int
    maxtime::T
    feval::Int
    final::Vector{T}
    fmin::T
    optimizer::Symbol
    returnvalue::Symbol
    nAGQ::Integer           # don't really belong here but I needed a place to store them
    REML::Bool
    sigma::Union{T,Nothing}
    fitlog::Vector{Tuple{Vector{T},T}} # not SVector because we would need to parameterize on size (which breaks GLMM)
end

function OptSummary(
    initial::Vector{T},
    lowerbd::Vector{T},
    optimizer::Symbol;
    ftol_rel::T=zero(T),
    ftol_abs::T=zero(T),
    xtol_rel::T=zero(T),
    xtol_abs::Vector{T}=zero(initial) .+ 1e-10,
    initial_step::Vector{T}=T[],
    maxfeval=-1,
    maxtime=T(-1),
) where {T<:AbstractFloat}
    fitlog = [(initial, T(Inf))]

    return OptSummary(
        initial,
        lowerbd,
        T(Inf),
        ftol_rel,
        ftol_abs,
        xtol_rel,
        xtol_abs,
        initial_step,
        maxfeval,
        maxtime,
        -1,
        copy(initial),
        T(Inf),
        optimizer,
        :FAILURE,
        1,
        false,
        nothing,
        fitlog,
    )
end

"""
    columntable(s::OptSummary, [stack::Bool=false])

Return `s.fitlog` as a `Tables.columntable`.

When `stack` is false (the default), there will be 3 columns in the result:
- `iter`: the sample number
- `objective`: the value of the objective at that sample
- `θ`: the parameter vector at that sample

(The term `sample` here refers to the fact that when the `thin` argument to the `fit` or
`refit!` call is greater than 1 only a subset of the iterations have results recorded.)

When `stack` is true, there will be 4 columns: `iter`, `objective`, `par`, and `value`
where `value` is the stacked contents of the `θ` vectors (the equivalent of `vcat(θ...)`)
and `par` is a vector of parameter numbers.
"""
function Tables.columntable(s::OptSummary; stack::Bool=false)
    fitlog = s.fitlog
    val = (; iter=axes(fitlog, 1), objective=last.(fitlog), θ=first.(fitlog))
    stack || return val
    θ1 = first(val.θ)
    k = length(θ1)
    return (;
        iter=repeat(val.iter; inner=k),
        objective=repeat(val.objective; inner=k),
        par=repeat(1:k; outer=length(fitlog)),
        value=foldl(vcat, val.θ; init=(eltype(θ1))[]),
    )
end

function Base.show(io::IO, ::MIME"text/plain", s::OptSummary)
    println(io, "Initial parameter vector: ", s.initial)
    println(io, "Initial objective value:  ", s.finitial)
    println(io)
    println(io, "Optimizer (from NLopt):   ", s.optimizer)
    println(io, "Lower bounds:             ", s.lowerbd)
    println(io, "ftol_rel:                 ", s.ftol_rel)
    println(io, "ftol_abs:                 ", s.ftol_abs)
    println(io, "xtol_rel:                 ", s.xtol_rel)
    println(io, "xtol_abs:                 ", s.xtol_abs)
    println(io, "initial_step:             ", s.initial_step)
    println(io, "maxfeval:                 ", s.maxfeval)
    println(io, "maxtime:                  ", s.maxtime)
    println(io)
    println(io, "Function evaluations:     ", s.feval)
    println(io, "Final parameter vector:   ", s.final)
    println(io, "Final objective value:    ", s.fmin)
    return println(io, "Return code:              ", s.returnvalue)
end

Base.show(io::IO, s::OptSummary) = Base.show(io, MIME"text/plain"(), s)

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

StructTypes.StructType(::Type{<:OptSummary}) = StructTypes.Mutable()
StructTypes.excludes(::Type{<:OptSummary}) = (:lowerbd,)

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
