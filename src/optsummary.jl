"""
    OptSummary

Summary of an `NLopt` optimization

# Fields

* `initial`: a copy of the initial parameter values in the optimization
* `lowerbd`: lower bounds on the parameter values
* `ftol_rel`: as in NLopt
* `ftol_abs`: as in NLopt
* `xtol_rel`: as in NLopt
* `xtol_abs`: as in NLopt
* `initial_step`: as in NLopt
* `maxfeval`: as in NLopt
* `final`: a copy of the final parameter values from the optimization
* `fmin`: the final value of the objective
* `feval`: the number of function evaluations
* `optimizer`: the name of the optimizer used, as a `Symbol`
* `returnvalue`: the return value, as a `Symbol`
* `nAGQ`: number of adaptive Gauss-Hermite quadrature points in deviance evaluation for GLMMs
* `REML`: use the REML criterion for LMM fits

The latter two fields are model characteristics and not related directly to the `NLopt` package or algorithms.
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
    final::Vector{T}
    fmin::T
    feval::Int
    optimizer::Symbol
    returnvalue::Symbol
    nAGQ::Integer           # don't really belong here but I needed a place to store them
    REML::Bool
end
function OptSummary(
    initial::Vector{T},
    lowerbd::Vector{T},
    optimizer::Symbol;
    ftol_rel::T = zero(T),
    ftol_abs::T = zero(T),
    xtol_rel::T = zero(T),
    initial_step::Vector{T} = T[],
) where {T<:AbstractFloat}
    OptSummary(
        initial,
        lowerbd,
        T(Inf),
        ftol_rel,
        ftol_abs,
        xtol_rel,
        zero(initial),
        initial_step,
        -1,
        copy(initial),
        T(Inf),
        -1,
        optimizer,
        :FAILURE,
        1,
        false,
    )
end

function Base.show(io::IO, s::OptSummary)
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
    println(io)
    println(io, "Function evaluations:     ", s.feval)
    println(io, "Final parameter vector:   ", s.final)
    println(io, "Final objective value:    ", s.fmin)
    println(io, "Return code:              ", s.returnvalue)
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
    if isempty(optsum.initial_step)
        optsum.initial_step = NLopt.initial_step(opt, optsum.initial, similar(lb))
    else
        NLopt.initial_step!(opt, optsum.initial_step)
    end
    opt
end
