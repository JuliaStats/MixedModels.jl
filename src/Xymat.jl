"""
    FeTerm{T,S}

Term with an explicit, constant matrix representation

Typically, an `FeTerm` represents the model matrix for the fixed effects.

!!! note
    `FeTerm` is not the same as [`FeMat`](@ref)!

# Fields
* `x`: full model matrix
* `piv`: pivot `Vector{Int}` for moving linearly dependent columns to the right
* `rank`: computational rank of `x`
* `cnames`: vector of column names
"""
struct FeTerm{T,S<:AbstractMatrix}
    x::S
    piv::Vector{Int}
    rank::Int
    cnames::Vector{String}
end

"""
    FeTerm(X::AbstractMatrix, cnms)

Convenience constructor for [`FeTerm`](@ref) that computes the rank and pivot with unit weights.

See the vignette "[Rank deficiency in mixed-effects models](@ref)" for more information on the
computation of the rank and pivot.
"""
function FeTerm(X::AbstractMatrix{T}, cnms) where {T}
    if iszero(size(X, 2))
        return FeTerm{T,typeof(X)}(X, Int[], 0, cnms)
    end
    rank, pivot = statsrank(X)
    # single-column rank deficiency is the result of a constant column vector
    # this generally happens when constructing a dummy response, so we don't
    # warn.
    if rank < length(pivot) && size(X, 2) > 1
        @warn "Fixed-effects matrix is rank deficient"
    end
    return FeTerm{T,typeof(X)}(X[:, pivot], pivot, rank, cnms[pivot])
end

"""
    FeTerm(X::SparseMatrixCSC, cnms)

Convenience constructor for a sparse [`FeTerm`](@ref) assuming full rank, identity pivot and unit weights.

Note: automatic rank deficiency handling may be added to this method in the future, as discussed in
the vignette "[Rank deficiency in mixed-effects models](@ref)" for general `FeTerm`.
"""
function FeTerm(X::SparseMatrixCSC, cnms::AbstractVector{String})
    #@debug "Full rank is assumed for sparse fixed-effect matrices."
    rank = size(X, 2)
    return FeTerm{eltype(X),typeof(X)}(X, collect(1:rank), rank, collect(cnms))
end

Base.copyto!(A::FeTerm{T}, src::AbstractVecOrMat{T}) where {T} = copyto!(A.x, src)

Base.eltype(::FeTerm{T}) where {T} = T

"""
    pivot(m::MixedModel)
    pivot(A::FeTerm)

Return the pivot associated with the FeTerm.
"""
@inline pivot(m::MixedModel) = pivot(m.feterm)
@inline pivot(A::FeTerm) = A.piv

function fullrankx(A::FeTerm)
    x, rnk = A.x, A.rank
    return rnk == size(x, 2) ? x : view(x, :, 1:rnk)  # this handles the zero-columns case
end

fullrankx(m::MixedModel) = fullrankx(m.feterm)

LinearAlgebra.rank(A::FeTerm) = A.rank

"""
    isfullrank(A::FeTerm)

Does `A` have full column rank?
"""
isfullrank(A::FeTerm) = A.rank == length(A.piv)

"""
    FeMat{T,S}

A matrix and a (possibly) weighted copy of itself.


Typically, an `FeMat` represents the fixed-effects model matrix with the response (`y`) concatenated as a final column.

!!! note
    `FeMat` is not the same as [`FeTerm`](@ref).

# Fields
- `xy`: original matrix, called `xy` b/c in practice this is `hcat(fullrank(X), y)`
- `wtxy`: (possibly) weighted copy of `xy` (shares storage with `xy` until weights are applied)

Upon construction the `xy` and `wtxy` fields refer to the same matrix
"""
mutable struct FeMat{T,S<:AbstractMatrix} <: AbstractMatrix{T}
    xy::S
    wtxy::S
end

function FeMat(A::FeTerm{T}, y::AbstractVector{T}) where {T}
    xy = hcat(fullrankx(A), y)
    return FeMat{T,typeof(xy)}(xy, xy)
end

Base.adjoint(A::FeMat) = Adjoint(A)

Base.eltype(::FeMat{T}) where {T} = T

Base.getindex(A::FeMat, i, j) = getindex(A.xy, i, j)

Base.length(A::FeMat) = length(A.xy)

function Base.:(*)(adjA::Adjoint{T,<:FeMat{T}}, B::FeMat{T}) where {T}
    return adjoint(adjA.parent.wtxy) * B.wtxy
end

function LinearAlgebra.mul!(
    R::StridedVecOrMat{T}, A::FeMat{T}, B::StridedVecOrMat{T}
) where {T}
    return mul!(R, A.wtxy, B)
end

function LinearAlgebra.mul!(C, adjA::Adjoint{T,<:FeMat{T}}, B::FeMat{T}) where {T}
    return mul!(C, adjoint(adjA.parent.wtxy), B.wtxy)
end

function reweight!(A::FeMat{T}, sqrtwts::Vector{T}) where {T}
    if !isempty(sqrtwts)
        if A.xy === A.wtxy
            A.wtxy = similar(A.xy)
        end
        mul!(A.wtxy, Diagonal(sqrtwts), A.xy)
    end
    return A
end

Base.size(A::FeMat) = size(A.xy)

Base.size(A::FeMat, i::Integer) = size(A.xy, i)
