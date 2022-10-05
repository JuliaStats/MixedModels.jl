"""
    UniformBlockDiagonal{T}

Homogeneous block diagonal matrices.  `k` diagonal blocks each of size `m×m`
"""
struct UniformBlockDiagonal{T} <: AbstractMatrix{T}
    data::Array{T,3}
end

function Base.axes(A::UniformBlockDiagonal)
    m, n, l = size(A.data)
    return (Base.OneTo(m * l), Base.OneTo(n * l))
end

function Base.copyto!(dest::Matrix{T}, src::UniformBlockDiagonal{T}) where {T}
    size(dest) == size(src) || throw(DimensionMismatch(""))
    fill!(dest, zero(T))
    sdat = src.data
    m, n, l = size(sdat)
    @inbounds for k in axes(sdat, 3)
        ioffset = (k - 1) * m
        joffset = (k - 1) * n
        for j in axes(sdat, 2)
            jind = joffset + j
            for i in axes(sdat, 1)
                dest[ioffset + i, jind] = sdat[i, j, k]
            end
        end
    end
    return dest
end

function Base.getindex(A::UniformBlockDiagonal{T}, i::Int, j::Int) where {T}
    @boundscheck checkbounds(A, i, j)
    Ad = A.data
    m, n, l = size(Ad)
    iblk, ioffset = divrem(i - 1, m)
    jblk, joffset = divrem(j - 1, n)
    return iblk == jblk ? Ad[ioffset + 1, joffset + 1, iblk + 1] : zero(T)
end

function LinearAlgebra.Matrix(A::UniformBlockDiagonal{T}) where {T}
    return copyto!(Matrix{T}(undef, size(A)), A)
end

function Base.size(A::UniformBlockDiagonal)
    m, n, l = size(A.data)
    return (l * m, l * n)
end

"""
    BlockedSparse{Tv,S,P}

A `SparseMatrixCSC` whose nonzeros form blocks of rows or columns or both.

# Members
* `cscmat`: `SparseMatrixCSC{Tv, Int32}` representation for general calculations
* `nzasmat`: nonzeros of `cscmat` as a dense matrix
* `colblkptr`: pattern of blocks of columns

The only time these are created are as products of `ReMat`s.
"""
mutable struct BlockedSparse{T,S,P} <: AbstractMatrix{T}
    cscmat::SparseMatrixCSC{T,Int32}
    nzsasmat::Matrix{T}
    colblkptr::Vector{Int32}
end

function densify(A::BlockedSparse, threshold::Real=0.1)
    m, n = size(A)
    if nnz(A) / (m * n) ≤ threshold
        A
    else
        Array(A)
    end
end

Base.size(A::BlockedSparse) = size(A.cscmat)

Base.size(A::BlockedSparse, d) = size(A.cscmat, d)

Base.getindex(A::BlockedSparse, i::Integer, j::Integer) = getindex(A.cscmat, i, j)

LinearAlgebra.Matrix(A::BlockedSparse) = Matrix(A.cscmat)

SparseArrays.sparse(A::BlockedSparse) = A.cscmat

SparseArrays.nnz(A::BlockedSparse) = nnz(A.cscmat)

function Base.copyto!(L::BlockedSparse{T}, A::SparseMatrixCSC{T}) where {T}
    size(L) == size(A) && nnz(L) == nnz(A) ||
        throw(DimensionMismatch("size(L) ≠ size(A) or nnz(L) ≠ nnz(A"))
    copyto!(nonzeros(L.cscmat), nonzeros(A))
    return L
end

LinearAlgebra.rdiv!(A::BlockedSparse, B::Diagonal) = rdiv!(A.cscmat, B)

function LinearAlgebra.mul!(
    C::BlockedSparse{T,1,P},
    A::SparseMatrixCSC{T,Ti},
    adjB::Adjoint{T,BlockedSparse{T,P,1}},
    α,
    β,
) where {T,P,Ti}
    return mul!(C.cscmat, A, adjoint(adjB.parent.cscmat), α, β)
end
