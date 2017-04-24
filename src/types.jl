"""
    HeteroBlkdMatrix

A matrix composed of heterogenous blocks.  Blocks can be sparse, dense or
diagonal.
"""

immutable HeteroBlkdMatrix <: AbstractMatrix{AbstractMatrix}
    blocks::Matrix{AbstractMatrix}
end
Base.size(A::HeteroBlkdMatrix) = size(A.blocks)
Base.getindex(A::HeteroBlkdMatrix, i::Int) = A.blocks[i]
Base.setindex!(A::HeteroBlkdMatrix, X, i::Integer) = setindex!(A.blocks, X, i)
@compat Base.IndexStyle(A::HeteroBlkdMatrix) = IndexLinear()

immutable Identity{T<:AbstractFloat} end

"""
    T<:AbstractFloat}

A `LowerTriangular{T, Matrix{T}}` and an integer vector, `mask`, of the potential non-zero elements.

In linear algebra operations an object `A` of this type acts like `I ⊗ A`,
for a suitably sized `I`.  These are the pattern matrices for blocks of `Λ`.
"""
immutable MaskedLowerTri{T<:AbstractFloat}
    m::LowerTriangular{T,Matrix{T}}
    mask::Vector{Int}
end
function MaskedLowerTri(v::Vector, T::DataType)
    n = sum(v)
    inds = reshape(1:abs2(n), (n, n))
    offset = 0
    mask = sizehint!(Int[], (n * (n + 1)) >> 1)
    for k in v
        for j in 1:k, i in j:k
            push!(mask, inds[offset + i, offset + j])
        end
        offset += k
    end
    MaskedLowerTri(LowerTriangular(eye(T, n)), mask)
end

=={T}(A::MaskedLowerTri{T}, B::MaskedLowerTri{T}) = A.m == B.m && A.mask == B.mask

"""
    LambdaTypes{T<:AbstractFloat}

Union of possible types in the `Λ` member of `[LinearMixedModel](@ref)`

These types are `Identity{T}`, `MaskedLowerTri{T}`, and `UniformScaling{T}`
"""
@compat const LambdaTypes{T} = Union{Identity{T}, MaskedLowerTri{T}, UniformScaling{T}}

"""
    OptSummary

Summary of an `NLopt` optimization

# Members
* `initial`: a copy of the initial parameter values in the optimization
* `lowerbd`: lower bounds on the parameter values
* `ftol_rel`: as in NLopt
* `ftol_abs`: as in NLopt
* `xtol_rel`: as in NLopt
* `xtol_abs`: as in NLopt
* `initial_step`: as in NLopt
* `final`: a copy of the final parameter values from the optimization
* `fmin`: the final value of the objective
* `feval`: the number of function evaluations
* `optimizer`: the name of the optimizer used, as a `Symbol`
* `returnvalue`: the return value, as a `Symbol`
"""
type OptSummary{T <: AbstractFloat}
    initial::Vector{T}
    lowerbd::Vector{T}
    finitial::T
    ftol_rel::T
    ftol_abs::T
    xtol_rel::T
    xtol_abs::Vector{T}
    initial_step::Vector{T}
    final::Vector{T}
    fmin::T
    feval::Int
    optimizer::Symbol
    returnvalue::Symbol
end
function OptSummary{T<:AbstractFloat}(initial::Vector{Float64}, lowerbd::Vector{T},
    optimizer::Symbol; ftol_rel::T=zero(T), ftol_abs::T=zero(T), xtol_rel::T=zero(T),
    initial_step::Vector{T}=T[])
    OptSummary(initial, lowerbd, T(Inf), ftol_rel, ftol_abs, xtol_rel, zeros(initial),
        initial_step, copy(initial), T(Inf), -1, optimizer, :FAILURE)
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
    NLopt.xtol_abs!(opt, optsum.xtol_abs) # absolute criterion on parameter values
    NLopt.lower_bounds!(opt, lb)
    NLopt.maxeval!(opt, optsum.feval)
    if isempty(optsum.initial_step)
        optsum.initial_step = NLopt.initial_step(opt, optsum.initial, similar(lb))
    else
        NLopt.initial_step!(opt, optsum.initial_step)
    end
    opt
end

@compat const AbstractFactor{V,R} = Union{NullableCategoricalVector{V,R},CategoricalVector{V,R},PooledDataVector{V,R}}

"""
    ReMat

Representation of the model matrix for random-effects terms

# Members
* `f`: the grouping factor as an `AbstractFactor`
* `z`: the transposed raw random-effects model matrix
* `fnm`: the name of the grouping factor as a `Symbol`
* `cnms`: a `Vector` of column names (row names after transposition) of `z`
"""
immutable ReMat{T<:AbstractFloat,V,R}
    f::AbstractFactor{V,R}
    z::Matrix{T}
    fnm::Symbol
    cnms::Vector
end

@compat abstract type MixedModel <: RegressionModel end # model with fixed and random effects

"""
    LinearMixedModel

Linear mixed-effects model representation

# Members
* `formula`: the formula for the model
* `fixefnames`: names of the fixed effects (for displaying coefficients)
* `wttrms`: a length `nt` vector of weighted model matrices. The last two elements are `X` and `y`.
* `trms`: a vector of unweighted model matrices.  If `isempty(sqrtwts)` the same object as `wttrms`
* `Λ`: a length `nt - 2` vector of lower triangular matrices
* `sqrtwts`: the `Diagonal` matrix of the square roots of the case weights.  Allowed to be size 0
* `A`: an `nt × nt` symmetric matrix of matrices representing `hcat(Z,X,y)'hcat(Z,X,y)`
* `L`: a `nt × nt` matrix of matrices - the lower Cholesky factor of `Λ'AΛ+I`
* `opt`: an [`OptSummary`](@ref) object
"""
immutable LinearMixedModel{T <: AbstractFloat} <: MixedModel
    formula::Formula
    fixefnames::Vector{String}
    wttrms::Vector
    trms::Vector
    sqrtwts::Diagonal{T}
    Λ::Vector
    A::Hermitian # cross-product blocks
    L::LowerTriangular
    optsum::OptSummary{T}
end

"""
    GeneralizedLinearMixedModel

Generalized linear mixed-effects model representation

Members:

- `LMM`: a [`LinearMixedModel`](@ref) - the local approximation to the GLMM.
- `β`: the fixed-effects vector
- `β₀`: similar to `β`. User in the PIRLS algorithm if step-halving is needed.
- `θ`: covariance parameter vector
- `b`: similar to `u`, equivalent to `broadcast!(*, b, LMM.Λ, u)`
- `u`: a vector of matrices of random effects
- `u₀`: similar to `u`.  Used in the PIRLS algorithm if step-halving is needed.
- `resp`: a `GlmResp` object
- `η`: the linear predictor
- `wt`: vector of prior case weights, a value of `T[]` indicates equal weights.
"""
immutable GeneralizedLinearMixedModel{T <: AbstractFloat} <: MixedModel
    LMM::LinearMixedModel{T}
    β::Vector{T}
    β₀::Vector{T}
    θ::Vector{T}
    b::Vector{Matrix{T}}
    u::Vector{Matrix{T}}
    u₀::Vector{Matrix{T}}
    resp::GlmResp
    η::Vector{T}
    wt::Vector{T}
end

"""
    VarCorr

An encapsulation of information on the fitted random-effects
variance-covariance matrices.

# Members
* `σ`: a `Vector{Vector{T}}` of unscaled standard deviations
* `ρ`: a `Vector{Matrix{T}}` of correlation matrices
* `fnms`: a `Vector{Symbol}` of grouping factor names
* `cnms`: a `Vector{Vector{String}}` of column names
* `s`: the estimate of σ, the standard deviation of the per-observation noise.  When there
is no scaling factor this value is `NaN`

The main purpose of defining this type is to isolate the logic in the show method.
"""
immutable VarCorr
    σ::Vector{Vector}
    ρ::Vector{Matrix}
    fnms::Vector{Symbol}
    cnms::Vector{Vector{String}}
    s
end
function VarCorr(m::MixedModel)
    LMM = lmm(m)
    Λ = LMM.Λ
    trms = LMM.trms
    fnms = Symbol[]
    cnms = Vector{String}[]
    T = eltype(Λ[1])
    σ, ρ = Vector{T}[], Matrix{T}[]
    ## FIXME Clean this up by mapping extractor functions
    for i in eachindex(Λ)
        λ = Λ[i]
        if !isa(λ, Identity)
            σi, ρi = stddevcor(λ)
            push!(σ, σi)
            push!(ρ, ρi)
            trmi = trms[i]
            push!(fnms, trmi.fnm)
            push!(cnms, trmi.cnms)
        end
    end
    VarCorr(σ, ρ, fnms, cnms, sdest(m))
end

function Base.show(io::IO, vc::VarCorr)
    # FIXME: Do this one term at a time
    fnms = vc.fnms
    stdm = vc.σ
    cor = vc.ρ
    cnms = reduce(vcat, vc.cnms)
    if isfinite(vc.s)
        push!(fnms,"Residual")
        push!(stdm, [1.])
        scale!(stdm, vc.s)
        push!(cnms, "")
    end
    nmwd = maximum(map(strwidth, string.(fnms))) + 1
    write(io, "Variance components:\n")
    cnmwd = max(6, maximum(map(strwidth, cnms))) + 1
    tt = vcat(stdm...)
    vars = showoff(abs2.(tt), :plain)
    stds = showoff(tt, :plain)
    varwd = 1 + max(length("Variance"), maximum(map(strwidth, vars)))
    stdwd = 1 + max(length("Std.Dev."), maximum(map(strwidth, stds)))
    write(io, " "^(2+nmwd))
    write(io, Base.cpad("Column", cnmwd))
    write(io, Base.cpad("Variance", varwd))
    write(io, Base.cpad("Std.Dev.", stdwd))
    any(s -> length(s) > 1, stdm) && write(io,"  Corr.")
    println(io)
    ind = 1
    for i in 1:length(fnms)
        stdmi = stdm[i]
        write(io, ' ')
        write(io, rpad(fnms[i], nmwd))
        write(io, rpad(cnms[ind], cnmwd))
        write(io, lpad(vars[ind], varwd))
        write(io, lpad(stds[ind], stdwd))
        ind += 1
        println(io)
        for j in 2:length(stdmi)
            write(io, " "^(1 + nmwd))
            write(io, rpad(cnms[ind], cnmwd))
            write(io, lpad(vars[ind], varwd))
            write(io, lpad(stds[ind], stdwd))
            ind += 1
            for k in 1:(j-1)
                @printf(io, "%6.2f", cor[i][j, k])
            end
            println(io)
        end
    end
end
