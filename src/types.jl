using StaticArrays, SparseArrays, LinearAlgebra

"""
    UniformBlockDiagonal{T}

Homogeneous block diagonal matrices.  `k` diagonal blocks each of size `m×m`
"""
struct UniformBlockDiagonal{T} <: AbstractMatrix{T}
    data::Array{T, 3}
    facevec::Vector{SubArray{T,2,Array{T,3}}}
end

function UniformBlockDiagonal(dat::Array{T,3}) where {T}
    UniformBlockDiagonal(dat,
        SubArray{T,2,Array{T,3}}[view(dat,:,:,i) for i in 1:size(dat, 3)])
end

Base.copy(A::UniformBlockDiagonal) = UniformBlockDiagonal(copy(A.data))

function Base.size(A::UniformBlockDiagonal)
    m, n, l = size(A.data)
    (l * m, l * n)
end

function Base.getindex(A::UniformBlockDiagonal{T}, i::Int, j::Int) where {T}
    Ad = A.data
    m, n, l = size(Ad)
    (0 < i ≤ l * m && 0 < j ≤ l * n) ||
        throw(IndexError("attempt to access $(l*m) × $(l*n) array at index [$i, $j]"))
    iblk, ioffset = divrem(i - 1, m)
    jblk, joffset = divrem(j - 1, n)
    iblk == jblk ? Ad[ioffset+1, joffset+1, iblk+1] : zero(T)
end

function LinearAlgebra.Matrix(A::UniformBlockDiagonal{T}) where {T}
    Ad = A.data
    m, n, l = size(Ad)
    mat = zeros(T, (m*l, n*l))
    @inbounds for k = 0:(l-1)
        kp1 = k + 1
        km = k * m
        kn = k * n
        for j = 1:n
            knpj = kn + j
            for i = 1:m
                mat[km + i, knpj] = Ad[i, j, kp1]
            end
        end
    end
    mat
end

"""
    RepeatedBlockDiagonal{T}

A block diagonal matrix consisting of `k` blocks each of which is the same `m×m` `Matrix{T}`.

This is the form of the `Λ` matrix from a `VectorFactorReTerm`.
"""
struct RepeatedBlockDiagonal{T,S<:AbstractMatrix{T}} <: AbstractMatrix{T}
    data::S
    nblocks::Int

    function RepeatedBlockDiagonal{T,S}(data,nblocks) where {T,S<:AbstractMatrix{T}}
        new{T,S}(data, nblocks)
    end
end

function RepeatedBlockDiagonal(A::AbstractMatrix, nblocks::Integer)
    RepeatedBlockDiagonal{eltype(A), typeof(A)}(A, Int(nblocks))
end

function Base.size(A::RepeatedBlockDiagonal)
    m, n = size(A.data)
    nb = A.nblocks
    (m * nb, n * nb)
end

function Base.getindex(A::RepeatedBlockDiagonal{T}, i::Int, j::Int) where {T}
    m, n = size(A.data)
    nb = A.nblocks
    (0 < i ≤ nb * m && 0 < j ≤ nb * n) ||
        throw(IndexError("attempt to access $(nb*m) × $(nb*n) array at index [$i, $j]"))
    iblk, ioffset = divrem(i - 1, m)
    jblk, joffset = divrem(j - 1, n)
    iblk == jblk ? A.data[ioffset+1, joffset+1] : zero(T)
end

function LinearAlgebra.Matrix(A::RepeatedBlockDiagonal{T}) where T
    mat = zeros(T, size(A))
    Ad = A.data
    m, n = size(Ad)
    nb = A.nblocks
    for k = 0:(nb-1)
        km = k * m
        kn = k * n
        for j = 1:n
            knpj = kn + j
            for i = 1:m
                mat[km + i, knpj] = Ad[i, j]
            end
        end
    end
    mat
end

"""
    BlockedSparse{Tv, Ti}

A `SparseMatrixCSC` whose nonzeros form blocks of rows or columns or both.

# Members
* `cscmat`: `SparseMatrixCSC{Tv, Ti}` representation for general calculations
* `nzsasmat`: Matrix{Tv} `cscmat.nzval` as a matrix
* `rowblocks`: `Vector{Vector{SubArray{Tv,1,Vector{Tv}}}}` of row blocks of nonzeros
* `colblocks`: `Vector{StridedMatrix{Tv}}` of column blocks of nonzeros
"""
mutable struct BlockedSparse{Tv,Ti} <: AbstractMatrix{Tv}
    cscmat::SparseMatrixCSC{Tv,Ti}
    nzsasmat::Matrix{Tv}
    rowblocks::Vector{Vector{SubArray{Tv,1,Vector{Tv}}}}
    colblocks::Vector{StridedMatrix{Tv}}
end

Base.size(A::BlockedSparse) = size(A.cscmat)

Base.size(A::BlockedSparse, d) = size(A.cscmat, d)

Base.getindex(A::BlockedSparse, i::Integer, j::Integer) = getindex(A.cscmat, i, j)

LinearAlgebra.Matrix(A::BlockedSparse) = Matrix(A.cscmat)

SparseArrays.sparse(A::BlockedSparse) = A.cscmat

SparseArrays.nnz(A::BlockedSparse) = nnz(A.cscmat)

function Base.copyto!(L::BlockedSparse{T,I}, A::SparseMatrixCSC{T,I}) where {T,I}
    @argcheck(size(L) == size(A) && nnz(L) == nnz(A), DimensionMismatch)
    copyto!(nonzeros(L.cscmat), nonzeros(A))
    L
end

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

The latter field doesn't really belong here but it has to be in a mutable struct in case it is changed.
"""
mutable struct OptSummary{T <: AbstractFloat}
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
    nAGQ::Integer           # doesn't really belong here but I needed some place to store it
    REML::Bool              # similarly, just needed a place to store this information
end
function OptSummary(initial::Vector{T}, lowerbd::Vector{T},
    optimizer::Symbol; ftol_rel::T=zero(T), ftol_abs::T=zero(T), xtol_rel::T=zero(T),
    initial_step::Vector{T}=T[]) where T <: AbstractFloat
    OptSummary(initial, lowerbd, T(Inf), ftol_rel, ftol_abs, xtol_rel, zero(initial),
        initial_step, -1, copy(initial), T(Inf), -1, optimizer, :FAILURE, 1, false)
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
    if length(optsum.xtol_abs) == length(lb)  # not true for the second optimization in GLMM
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

abstract type MixedModel{T} <: RegressionModel end # model with fixed and random effects

"""
    LinearMixedModel

Linear mixed-effects model representation

## Fields

* `formula`: the formula for the model
* `trms`: a `Vector` of `AbstractTerm` types representing the model.  The last element is the response.
* `sqrtwts`: vector of square roots of the case weights.  Can be empty.
* `A`: an `nt × nt` symmetric `BlockMatrix` of matrices representing `hcat(Z,X,y)'hcat(Z,X,y)`
* `L`: a `nt × nt` `BlockMatrix` - the lower Cholesky factor of `Λ'AΛ+I`
* `optsum`: an [`OptSummary`](@ref) object

## Properties

* `θ` or `theta`: the covariance parameter vector used to form λ
* `β` or `beta`: the fixed-effects coefficient vector
* `λ` or `lambda`: a vector of lower triangular matrices repeated on the diagonal blocks of `Λ`
* `σ` or `sigma`: current value of the standard deviation of the per-observation noise
* `b`: random effects on the original scale, as a vector of matrices
* `u`: random effects on the orthogonal scale, as a vector of matrices
* `lowerbd`: lower bounds on the elements of θ
* `X`: the fixed-effects model matrix
* `y`: the response vector
"""
struct LinearMixedModel{T <: AbstractFloat} <: MixedModel{T}
    formula::Formula
    trms::Vector
    sqrtwts::Vector{T}
    A::BlockMatrix{T}            # cross-product blocks
    L::LowerTriangular{T,BlockArray{T,2,AbstractMatrix{T}}}
    optsum::OptSummary{T}
end

function normalized_variance_cumsum(Λ)
    vars = cumsum(abs2.(svdvals(Λ)))
    vars ./ vars[end]
end

function Base.getproperty(m::LinearMixedModel, s::Symbol)
    if s ∈ (:θ, :theta)
        getθ(m)
    elseif s ∈ (:β, :beta)
        fixef(m)
    elseif s ∈ (:λ, :lambda)
        getΛ(m)
    elseif s ∈ (:σ, :sigma)
        sdest(m)
    elseif s == :b
        ranef(m)
    elseif s == :u
        ranef(m, uscale = true)
    elseif s == :lowerbd
        m.optsum.lowerbd
    elseif s == :X
        m.trms[end - 1].x
    elseif s == :y
        vec(m.trms[end].x)
    elseif s == :rePCA
        normalized_variance_cumsum.(getΛ(m))
    else
        getfield(m, s)
    end
end

Base.setproperty!(m::LinearMixedModel, s::Symbol, y) = s == :θ ? setθ!(m, y) : setfield!(m, s, y)

Base.propertynames(m::LinearMixedModel, private=false) =
    (:formula, :trms, :A, :L, :optsum, :θ, :theta, :β, :beta, :λ, :lambda, :σ, :sigma, :b, :u, :lowerbd, :X, :y, :rePCA)

struct RaggedArray{T,I}
    vals::Vector{T}
    inds::Vector{I}
end
function Base.sum!(s::AbstractVector{T}, a::RaggedArray{T}) where T
    for (v, i) in zip(a.vals, a.inds)
        s[i] += v
    end
    s
end

"""
    GeneralizedLinearMixedModel

Generalized linear mixed-effects model representation

# Fields
- `LMM`: a [`LinearMixedModel`](@ref) - the local approximation to the GLMM.
- `β`: the fixed-effects vector
- `β₀`: similar to `β`. Used in the PIRLS algorithm if step-halving is needed.
- `θ`: covariance parameter vector
- `b`: similar to `u`, equivalent to `broadcast!(*, b, LMM.Λ, u)`
- `u`: a vector of matrices of random effects
- `u₀`: similar to `u`.  Used in the PIRLS algorithm if step-halving is needed.
- `resp`: a `GlmResp` object
- `η`: the linear predictor
- `wt`: vector of prior case weights, a value of `T[]` indicates equal weights.
The following fields are used in adaptive Gauss-Hermite quadrature, which applies
only to models with a single random-effects term, in which case their lengths are
the number of levels in the grouping factor for that term.  Otherwise they are
zero-length vectors.
- `devc`: vector of deviance components
- `devc0`: vector of deviance components at offset of zero
- `sd`: approximate standard deviation of the conditional density
- `mult`: multiplier

# Properties

In addition to the fieldnames, the following names are also accessible through the `.` extractor

- `theta`: synonym for `θ`
- `beta`: synonym for `β`
- `σ` or `sigma`: common scale parameter (value is `NaN` for distributions without a scale parameter)
- `lowerbd`: vector of lower bounds on the combined elements of `β` and `θ`
- `formula`, `trms`, `A`, `L`, and `optsum`: fields of the `LMM` field
- `X`: fixed-effects model matrix
- `y`: response vector

"""
struct GeneralizedLinearMixedModel{T <: AbstractFloat} <: MixedModel{T}
    LMM::LinearMixedModel{T}
    β::Vector{T}
    β₀::Vector{T}
    θ::Vector{T}
    b::Vector{Matrix{T}}
    u::Vector{Matrix{T}}
    u₀::Vector{Matrix{T}}
    resp::GLM.GlmResp
    η::Vector{T}
    wt::Vector{T}
    devc::Vector{T}
    devc0::Vector{T}
    sd::Vector{T}
    mult::Vector{T}
end


function Base.getproperty(m::GeneralizedLinearMixedModel, s::Symbol)
    if s == :theta
        m.θ
    elseif s == :beta
        m.β
    elseif s ∈ (:λ, :lambda)
        getΛ(m)
    elseif s ∈ (:σ, :sigma)
        sdest(m)
    elseif s == :lowerbd
        m.LMM.optsum.lowerbd
    elseif s ∈ (:formula, :trms, :A, :L, :optsum)
        getfield(m.LMM, s)
    elseif s == :X
        m.LMM.trms[end - 1].x
    elseif s == :y
        vec(m.LMM.trms[end].x)
    else
        getfield(m, s)
    end
end

function Base.setproperty!(m::GeneralizedLinearMixedModel, s::Symbol, y)
    if s ∈ (:θ, :theta)
        setθ!(m, y)
    elseif s ∈ (:β, :beta)
        setβ!(m, y)
    elseif s ∈ (:βθ, :betatheta)
        setβθ!(m, y)
    else
        setfield!(m, s, y)
    end
end

Base.propertynames(m::GeneralizedLinearMixedModel, private=false) =
    (:theta, :beta, :λ, :lambda, :σ, :sigma, :X, :y, :lowerbd, fieldnames(typeof(m))..., fieldnames(typeof(m.LMM))...)

"""
    VarCorr

An encapsulation of information on the fitted random-effects
variance-covariance matrices.

# Members
* `σ`: a `Vector{Vector{T}}` of unscaled standard deviations
* `ρ`: a `Vector{Matrix{T}}` of correlation matrices
* `fnms`: a `Vector{Symbol}` of grouping factor names
* `cnms`: a `Vector{Vector{String}}` of column names
* `s`: the estimate of σ, the standard deviation of the per-observation noise.  When there is no scaling factor this value is `NaN`

The main purpose of defining this type is to isolate the logic in the show method.
"""
struct VarCorr{T}
    σ::Vector{Vector{T}}
    ρ::Vector{Matrix{T}}
    fnms::Vector{Symbol}
    cnms::Vector{Vector{String}}
    s::T
end
function VarCorr(m::MixedModel{T}) where T
    fnms = Symbol[]
    cnms = Vector{String}[]
    σ = Vector{T}[]
    ρ = Matrix{T}[]
    for trm in reterms(m)
        σi, ρi = stddevcor(trm)
        push!(σ, σi)
        push!(ρ, ρi)
        push!(fnms, trm.fnm)
        push!(cnms, trm.cnms)
    end
    VarCorr(σ, ρ, fnms, cnms, sdest(m))
end

cpad(s::String, n::Integer) = rpad(lpad(s, (n + textwidth(s)) >> 1), n)

function Base.show(io::IO, vc::VarCorr)
    # FIXME: Do this one term at a time
    fnms = copy(vc.fnms)
    stdm = copy(vc.σ)
    cor = vc.ρ
    cnms = reduce(append!, vc.cnms, init=String[])
    if isfinite(vc.s)
        push!(fnms, :Residual)
        push!(stdm, [1.])
        rmul!(stdm, vc.s)
        push!(cnms, "")
    end
    nmwd = maximum(map(textwidth, string.(fnms))) + 1
    write(io, "Variance components:\n")
    cnmwd = max(6, maximum(map(textwidth, cnms))) + 1
    tt = vcat(stdm...)
    vars = showoff(abs2.(tt), :plain)
    stds = showoff(tt, :plain)
    varwd = 1 + max(length("Variance"), maximum(map(textwidth, vars)))
    stdwd = 1 + max(length("Std.Dev."), maximum(map(textwidth, stds)))
    write(io, " "^(2+nmwd))
    write(io, cpad("Column", cnmwd))
    write(io, cpad("Variance", varwd))
    write(io, cpad("Std.Dev.", stdwd))
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
