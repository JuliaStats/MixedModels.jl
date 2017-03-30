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
    MaskedMatrix

A matrix, `m`, and an integer vector, `mask`, of the potential non-zero elements.

Within this package all the `MaskedMatrix` objects will be LowerTriangular, but
that is not a requirement.

In linear algebra operations an object `A` of this type acts like `I ⊗ A`,
for a suitably sized `I`.  These are the pattern matrices for blocks of `Λ` corresponding
to a [`VectorReMat`](@ref).
"""
struct MaskedMatrix{T<:AbstractFloat}
    m::AbstractMatrix{T}
    mask::Vector{Int}
end
## FIXME: This section is too slapdash.  There is no guarantee that a MaskedMatrix is
## LowerTriangular
cond(A::MaskedMatrix) = cond(LowerTriangular(A.m))
