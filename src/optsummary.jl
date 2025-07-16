"""
    OptSummary

Summary of an optimization

# Fields

## Tolerances, initial and final values
* `initial`: a copy of the initial parameter values in the optimization
* `finitial`: the initial value of the objective
* `lowerbd`: lower bounds on the parameter values
* `final`: a copy of the final parameter values from the optimization
* `fmin`: the final value of the objective
* `feval`: the number of function evaluations
   Available backends can be examined via `OPTIMIZATION_BACKENDS`.
* `returnvalue`: the return value, as a `Symbol`. The available return values will differ between backends.
* `xtol_zero_abs`: the tolerance for a near zero parameter to be considered practically zero
* `ftol_zero_abs`: the tolerance for change in the objective for setting a near zero parameter to zero
* `maxfeval`: as in NLopt (`maxeval`) and PRIMA (`maxfun`)

## Choice of optimizer and backend
* `optimizer`: the name of the optimizer used, as a `Symbol`
* `backend`: the optimization library providing the optimizer, default is `NLoptBackend`.

## Backend-specific fields
* `ftol_rel`: as in NLopt, not used in PRIMA
* `ftol_abs`: as in NLopt, not used in PRIMA
* `xtol_rel`: as in NLopt, not used in PRIMA
* `xtol_abs`: as in NLopt, not used in PRIMA
* `initial_step`: as in NLopt, not used in PRIMA
* `maxtime`: as in NLopt, not used in PRIMA
* `rhobeg`: as in PRIMA, not used in NLopt
* `rhoend`: as in PRIMA, not used in NLopt

## MixedModels-specific fields, unrelated to the optimizer
* `fitlog`: a `Vector{T}` recording the objective values and parameter vectors from the optimization
* `nAGQ`: number of adaptive Gauss-Hermite quadrature points in deviance evaluation for GLMMs
* `REML`: use the REML criterion for LMM fits
* `sigma`: a priori value for the residual standard deviation for LMM

!!! note
    The internal storage of the parameter values within `fitlog` may change in
    the future to use a different subtype of `AbstractVector` (e.g., `StaticArrays.SVector`)
    for each snapshot without being considered a breaking change.

!!! note
    The exact order and number of fields may change as support for additional backends and features
    thereof are added. In other words: use the keyword constructor and do **not** use the positional
    constructor.
"""
Base.@kwdef mutable struct OptSummary{T<:AbstractFloat}
    initial::Vector{T}
    finitial::T = Inf * one(eltype(initial))
    lowerbd::Vector{T}
    final::Vector{T} = copy(initial)
    fmin::T = Inf * one(eltype(initial))
    feval::Int = -1
    returnvalue::Symbol = :FAILURE
    xtol_zero_abs::T = eltype(initial)(0.001)
    ftol_zero_abs::T = eltype(initial)(1.e-5)
    maxfeval::Int = -1

    optimizer::Symbol = :newuoa
    backend::Symbol = :prima

    # the @kwdef macro isn't quite smart enough for us to use the type parameter
    # for the default values, but we can fake it
    ftol_rel::T = eltype(initial)(1.0e-12)
    ftol_abs::T = eltype(initial)(1.0e-8)
    xtol_rel::T = zero(eltype(initial))
    xtol_abs::Vector{T} = zero(initial) .+ 1e-10
    initial_step::Vector{T} = empty(initial)
    maxtime::T = -one(eltype(initial))

    rhobeg::T = one(T)
    rhoend::T = rhobeg / 1_000_000

    # not SVector because we would need to parameterize on size (which breaks GLMM)
    fitlog::Vector{T} = T[]
    nAGQ::Int = 1
    REML::Bool = false
    sigma::Union{T,Nothing} = nothing
end

function OptSummary(
    initial::Vector{T},
    lowerbd::Vector{S},
    optimizer::Symbol=:newuoa; kwargs...,
) where {T<:AbstractFloat,S<:AbstractFloat}
    TS = promote_type(T, S)
    return OptSummary{TS}(; initial, lowerbd, optimizer, kwargs...)
end

function Base.show(io::IO, ::MIME"text/plain", s::OptSummary)
    println(io, "Initial parameter vector: ", s.initial)
    println(io, "Initial objective value:  ", s.finitial)
    println(io)
    println(io, "Backend:                  ", s.backend)
    println(io, "Optimizer:                ", s.optimizer)
    println(io, "Lower bounds:             ", s.lowerbd)

    for param in opt_params(Val(s.backend))
        println(io, rpad(string(param, ":"), length("Initial parameter vector: ")),
            getfield(s, param))
    end
    println(io)
    println(io, "Function evaluations:     ", s.feval)
    println(io, "xtol_zero_abs:            ", s.xtol_zero_abs)
    println(io, "ftol_zero_abs:            ", s.ftol_zero_abs)
    println(io, "Final parameter vector:   ", s.final)
    println(io, "Final objective value:    ", s.fmin)
    println(io, "Return code:              ", s.returnvalue)
    return nothing
end

Base.show(io::IO, s::OptSummary) = Base.show(io, MIME("text/plain"), s)

function Base.:(==)(o1::OptSummary{T}, o2::OptSummary{T}) where {T}
    return all(fieldnames(OptSummary)) do fn
        return getfield(o1, fn) == getfield(o2, fn)
    end
end

"""
    OPTIMIZATION_BACKENDS

A list of currently available optimization backends.
"""
const OPTIMIZATION_BACKENDS = Symbol[]
optimize!(m::MixedModel; kwargs...) = optimize!(m, Val(m.optsum.backend); kwargs...)

"""
    optimize!(::LinearMixedModel, ::Val{backend}; kwargs...)
    optimize!(::GeneralizedLinearMixedModel, ::Val{backend}; kwargs...)

Perform optimization on a mixed model, minimizing the objective.

`optimize!` set ups the call to the backend optimizer using the options contained in the
model's `optsum` field. It then calls that optimizer and returns `xmin, fmin`.
Providing support for a new backend involves defining appropriate `optimize!` methods
with the second argument of type `::Val{:backend_name}` and adding `:backend_name` to
`OPTIMIZATION_BACKENDS`. Similarly, a method `opt_params(::Val{:backend_name})` should
be defined, which returns the optimization parameters (e.g. `xtol_abs` or `rho_end`) used
by the backend.

Common keyword arguments are `progress` to show a progress meter as well as
`nAQG` and `fast` for GLMMs.
"""
function optimize! end

"""
    opt_params(::Val{backend})

Return a collection of the fields of [`OptSummary`](@ref) used by backend.

`:xtol_zero_abs`, `:ftol_zero_abs` do not need to be specified because
they are used _after_ optimization and are thus shared across backends.
"""
function opt_params end
