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

function Base.copyto!(dest::UniformBlockDiagonal{T}, src::UniformBlockDiagonal{T}) where{T}
    sdat = src.data
    ddat = dest.data
    size(ddat) == size(sdat) || throw(DimensionMismatch(""))
    copyto!(ddat, sdat)
    dest
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

function Base.size(A::UniformBlockDiagonal)
    m, n, l = size(A.data)
    (l * m, l * n)
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
    size(L) == size(A) && nnz(L) == nnz(A) || throw(DimensionMismatch("size(L) ≠ size(A) or nnz(L) ≠ nnz(A"))
    copyto!(nonzeros(L.cscmat), nonzeros(A))
    L
end

function densify(A::BlockedSparse, threshold::Real = 0.25)
    m, n = size(A)
    nnz(A)/(m * n)  ≤ threshold ? A : Matrix(A)
end
