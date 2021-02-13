"""
    FeTerm{T,S}

Term with an explicit, constant matrix representation

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
    st = statsqr(X)
    pivot = st.p
    rank = findfirst(<=(0), diff(st.p))
    rank = isnothing(rank) ? length(pivot) : rank
    Xp = pivot == collect(1:size(X, 2)) ? X : X[:, pivot]
        # single-column rank deficiency is the result of a constant column vector
        # this generally happens when constructing a dummy response, so we don't
        # warn.
    if rank < length(pivot) && size(X,2) > 1
        @warn "Fixed-effects matrix is rank deficient"
    end
    FeTerm{T,typeof(X)}(Xp, pivot, rank, cnms[pivot])
end

"""
    FeTerm(X::SparseMatrixCSC, cnms)
    
Convenience constructor for a sparse [`FeTerm`](@ref) assuming full rank, identity pivot and unit weights.

Note: automatic rank deficiency handling may be added to this method in the future, as discused in
the vignette "[Rank deficiency in mixed-effects models](@ref)" for general `FeTerm`.
"""
function FeTerm(X::SparseMatrixCSC, cnms)
    @debug "Full rank is assumed for sparse fixed-effect matrices."
    rank = size(X,2)
    FeTerm{eltype(X),typeof(X)}(X, X, 1:rank, rank, cnms)
end

Base.copyto!(A::FeTerm{T}, src::AbstractVecOrMat{T}) where {T} = copyto!(A.x, src)

Base.eltype(::FeTerm{T}) where {T} = T

function fullrankx(A::FeTerm)
    x, rnk = A.x, A.rank
    rnk == size(x, 2) ? x : view(x, :, 1:rnk)  # this handles the zero-columns case
end

LinearAlgebra.rank(A::FeTerm) = A.rank

"""
    isfullrank(A::FeTerm)

Does `A` have full column rank?
"""
isfullrank(A::FeTerm) = A.rank == length(A.piv)

"""
    FeMat{T,S}

A matrix and a (possibly) weighted copy of itself.

# Fields
- `xy`: original matrix, called `xy` b/c in practice this is `hcat(fullrank(X), y)`
- `wtxy`: (possibly) weighted copy of `xy`

Upon construction the `xy` and `wtxy` fields refer to the same matrix
"""
mutable struct FeMat{T,S<:AbstractMatrix} <: AbstractMatrix{T}
    xy::S
    wtxy::S
end

function FeMat(A::FeTerm{T}, y::AbstractVector{T}) where {T}
    xy = hcat(fullrankx(A), y)
    FeMat{T,typeof(xy)}(xy, xy)
end

Base.adjoint(A::FeMat) = Adjoint(A)

Base.eltype(::FeMat{T}) where {T} = T

Base.length(A::FeMat) = length(A.wtxy)

function *(adjA::Adjoint{T,<:FeMat{T}}, B::FeMat{T}) where {T}
    adjA.parent.wtxy' * B.wtxy
end

function LinearAlgebra.mul!(R::StridedVecOrMat{T}, A::FeMat{T}, B::StridedVecOrMat{T}) where {T}
    mul!(R, A.wtxy, B)
end

function LinearAlgebra.mul!(C, adjA::Adjoint{T,<:FeMat{T}}, B::FeMat{T}) where {T}
    mul!(C, adjA.parent.wtxy', B.wtxy)
end

function reweight!(A::FeMat{T}, sqrtwts::Vector{T}) where {T}
    if !isempty(sqrtwts)
        if A.xy === A.wtxy
            A.wtxy = similar(A.xy)
        end
        mul!(A.wtxy, Diagonal(sqrtwts), A.xy)
    end
    A
end

Base.size(A::FeMat) = size(A.xy)

Base.size(A::FeMat, i::Integer) = size(A.xy, i)
