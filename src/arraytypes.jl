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
    size(L) == size(A) && nnz(L) == nnz(A) || throw(DimensionMismatch("size(L) ≠ size(A) or nnz(L) ≠ nnz(A"))
    copyto!(nonzeros(L.cscmat), nonzeros(A))
    L
end
