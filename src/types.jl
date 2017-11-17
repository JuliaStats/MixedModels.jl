"""
    UniformBlockDiagonal{T}

Homogeneous block diagonal matrices.  `k` diagonal blocks each of size `m×m`
"""
struct UniformBlockDiagonal{T} <: AbstractMatrix{T}
    data::Array{T, 3}
    facevec::Vector{SubArray{T,2,Array{T,3}}}
end

function UniformBlockDiagonal(dat::Array{T,3}) where T
    UniformBlockDiagonal(dat, 
        SubArray{T,2,Array{T,3}}[view(dat,:,:,i) for i in 1:size(dat, 3)])
end

function Base.size(A::UniformBlockDiagonal)
    m, n, l = size(A.data)
    (l * m, l * n)
end

function Base.getindex(A::UniformBlockDiagonal{T}, i::Int, j::Int) where {T}
    m, n, l = size(A.data)
    (0 < i ≤ l * m && 0 < j ≤ l * n) ||
        throw(IndexError("attempt to access $(l*m) × $(l*n) array at index [$i, $j]"))
    iblk, ioffset = divrem(i - 1, m)
    jblk, joffset = divrem(j - 1, n)
    if iblk == jblk
        A.data[ioffset+1, joffset+1, iblk+1]
    else
        zero(T)
    end
end

function Base.full(A::UniformBlockDiagonal{T}) where T
    res = zeros(T, size(A))
    Ad = A.data
    m, n, l = size(Ad)
    offseti = 0
    offsetj = 0
    for k = 1:l
        for j = 1:n, i = 1:m
            res[offseti + i, offsetj + j] = Ad[i, j, k]
        end
        offseti += m
        offsetj += n
    end
    res
end

"""
    BlockedSparse{T}

A `SparseMatrixCSC` whose nonzeros form blocks of rows or columns or both.

# Members
* `cscmat`: `SparseMatrixCSC{T}` representation for general calculations
* `nzsasmat`: ReshapedArray{T,2,SubArray{T, 1, Vector{T}}} `cscmat.nzval` as a matrix 
* `rowblocks`: `Vector{Vector{SubArray{T,2,Matrix{T}}}}` of row blocks of nonzeros
* `colblocks`: `Vector{ReshapedArray{T,2,SubArray{T,1,Vector{T}}}}` of column blocks of nonzeros
"""
mutable struct BlockedSparse{Tv,Ti} <: AbstractMatrix{Tv}
    cscmat::SparseMatrixCSC{Tv,Ti}
    nzsasmat::Matrix{Tv}
    rowblocks::Vector{Vector{SubArray{Tv,1,Vector{Tv}}}}
    colblocks::Vector{StridedMatrix{Tv}}
end

Base.size(A::BlockedSparse) = size(A.cscmat)
Base.size(A::BlockedSparse, d) = size(A.cscmat, d)
Base.getindex(A::BlockedSparse{T}, i::Integer, j::Integer) where {T} = getindex(A.cscmat, i, j)
Base.full(A::BlockedSparse{T}) where {T} = full(A.cscmat)
Base.sparse(A::BlockedSparse) = A.cscmat
Base.nnz(A::BlockedSparse) = nnz(A.cscmat)
function Base.copy!(L::BlockedSparse{T,I}, A::SparseMatrixCSC{T,I}) where {T,I}
    @argcheck(nnz(L) == nnz(A), DimensionMismatch)
    copy!(nonzeros(L.cscmat), nonzeros(A))
    L
end
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
* `maxfeval`: as in NLopt
* `final`: a copy of the final parameter values from the optimization
* `fmin`: the final value of the objective
* `feval`: the number of function evaluations
* `optimizer`: the name of the optimizer used, as a `Symbol`
* `returnvalue`: the return value, as a `Symbol`
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
end
function OptSummary(initial::Vector{T}, lowerbd::Vector{T},
    optimizer::Symbol; ftol_rel::T=zero(T), ftol_abs::T=zero(T), xtol_rel::T=zero(T),
    initial_step::Vector{T}=T[]) where T <: AbstractFloat
    OptSummary(initial, lowerbd, T(Inf), ftol_rel, ftol_abs, xtol_rel, zeros(initial),
        initial_step, -1, copy(initial), T(Inf), -1, optimizer, :FAILURE)
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

abstract type AbstractTerm{T} end

abstract type MixedModel{T} <: RegressionModel end # model with fixed and random effects

"""
    LinearMixedModel

Linear mixed-effects model representation

# Members
* `formula`: the formula for the model
* `trms`: a `Vector{AbstractTerm}` representing the model.  The last element is the response.
* `sqrtwts`: vector of square roots of the case weights.  Can be empty.
* `A`: an `nt × nt` symmetric `BlockMatrix` of matrices representing `hcat(Z,X,y)'hcat(Z,X,y)`
* `L`: a `nt × nt` `BlockMatrix` - the lower Cholesky factor of `Λ'AΛ+I`
* `optsum`: an [`OptSummary`](@ref) object
"""
struct LinearMixedModel{T <: AbstractFloat} <: MixedModel{T}
    formula::Formula
    trms::Vector{AbstractTerm{T}}
    sqrtwts::Vector{T}
    A::BlockMatrix{T}            # cross-product blocks
    L::LowerTriangular{T,BlockArray{T,2,AbstractMatrix{T}}}
    optsum::OptSummary{T}
end

"""
    GeneralizedLinearMixedModel

Generalized linear mixed-effects model representation

# Members
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
"""
struct GeneralizedLinearMixedModel{T <: AbstractFloat} <: MixedModel{T}
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

function Base.show(io::IO, vc::VarCorr)
    # FIXME: Do this one term at a time
    fnms = copy(vc.fnms)
    stdm = copy(vc.σ)
    cor = vc.ρ
    cnms = reduce(append!, String[], vc.cnms)
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
