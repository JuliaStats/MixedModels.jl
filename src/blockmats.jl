"""
    HeteroBlkdMatrix

A matrix composed of heterogenous blocks.  Blocks can be sparse, dense or
diagonal.
"""

struct HeteroBlkdMatrix <: AbstractMatrix{AbstractMatrix}
    blocks::Matrix{AbstractMatrix}
end
Base.size(A::HeteroBlkdMatrix) = size(A.blocks)
Base.getindex(A::HeteroBlkdMatrix, i::Int) = A.blocks[i]
Base.setindex!(A::HeteroBlkdMatrix, X, i::Integer) = setindex!(A.blocks, X, i)
Base.IndexStyle(A::HeteroBlkdMatrix) = Base.IndexLinear()

"""
    MaskedLowerTri{T<:AbstractFloat}

A `LowerTriangular{T, Matrix{T}}` and an integer vector, `mask`, of the potential non-zero elements.

In linear algebra operations an object `A` of this type acts like `I ⊗ A`,
for a suitably sized `I`.  These are the pattern matrices for blocks of `Λ`.
"""
struct MaskedLowerTri{T<:AbstractFloat}
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

cond(A::MaskedLowerTri) = cond(A.m)

nθ(A::MaskedLowerTri) = length(A.mask)

"""
    getθ!{T}(v::AbstractVector{T}, A::MaskedLowerTri{T})

Overwrite `v` with the elements of the blocks in the lower triangle of `A` (column-major ordering)
"""
function getθ!{T<:AbstractFloat}(v::StridedVector{T}, A::MaskedLowerTri{T})
    @argcheck length(v) == length(A.mask) DimensionMismatch
    mask = A.mask
    m = A.m.data
    @inbounds for i in eachindex(mask)
        v[i] = m[mask[i]]
    end
    v
end

"""
    getθ(A::MaskedLowerTri{T})

Return a vector of the elements of the lower triangle blocks in `A` (column-major ordering)
"""
getθ(A::MaskedLowerTri) = A.m.data[A.mask]
getθ(A::AbstractVector) = mapreduce(getθ, vcat, A)

"""
    lowerbd{T}(A::MaskedLowerTri{T})

Return the vector of lower bounds on the parameters, `θ`.

These are the elements in the lower triangle in column-major ordering.
Diagonals have a lower bound of `0`.  Off-diagonals have a lower-bound of `-Inf`.
"""
lowerbd{T}(v::Vector{MaskedLowerTri{T}}) = mapreduce(lowerbd, vcat, v)
lowerbd{T}(A::MaskedLowerTri{T}) = T[x ∈ diagind(A.m.data) ? zero(T) : convert(T, -Inf) for x in A.mask]

function setθ!{T}(A::MaskedLowerTri{T}, v::AbstractVector{T})
    @argcheck length(v) == length(A.mask) DimensionMismatch
    m = A.m.data
    mask = A.mask
    @inbounds for i in eachindex(mask)
        m[mask[i]] = v[i]
    end
    A
end
