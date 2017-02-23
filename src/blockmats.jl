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
Base.linearindexing(A::HeteroBlkdMatrix) = Base.LinearFast()

"""
    UniformSc

Like UniformScaling but allowing for a more general type T
"""
type UniformSc{T}
    λ::T
end

typealias UniformScLT{T} UniformSc{LowerTriangular{T, Matrix{T}}}
