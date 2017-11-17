"""
    rankUpdate!(A, C)
    rankUpdate!(α, A, C)
    rankUpdate!(α, A, β, C)

A rank-k update of a Hermitian (Symmetric) matrix.

`α` and `β` both default to 1.0.  When `α` is -1.0 this is a downdate operation.
The name `rankUpdate!` is borrowed from [https://github.com/andreasnoack/LinearAlgebra.jl]
"""
function rankUpdate! end

function rankUpdate!(α::T, a::StridedVector{T},
                     A::HermOrSym{T,S}) where {T<:BlasReal,S<:StridedMatrix}
    BLAS.syr!(A.uplo, α, a, A.data)
    A
end

rankUpdate!(a::StridedVector{T}, A::HermOrSym{T,S}) where {T<:BlasReal,S<:StridedMatrix} =
    rankUpdate!(one(T), a, A)

rankUpdate!(α::T, A::StridedMatrix{T}, β::T,
            C::HermOrSym{T,S}) where {T<:BlasReal,S<:StridedMatrix} =
    BLAS.syrk!(C.uplo, 'N', α, A, β, C.data)

rankUpdate!(α::T, A::StridedMatrix{T}, C::HermOrSym{T,S}) where {T<:Real,S<:StridedMatrix} =
    rankUpdate!(α, A, one(T), C)

rankUpdate!(A::StridedMatrix{T}, C::HermOrSym{T,S}) where {T<:Real,S<:StridedMatrix} =
    rankUpdate!(one(T), A, one(T), C)

function rankUpdate!(α::T, A::SparseMatrixCSC{T},
                     β::T, C::HermOrSym{T,S}) where {T,S<:StridedMatrix{T}}
    m, n = size(A)
    @argcheck m == size(C, 2) && C.uplo == 'L' DimensionMismatch
    Cd = C.data
    β == 1 || scale!(LowerTriangular(Cd), β)
    rv = rowvals(A)
    nz = nonzeros(A)
    @inbounds for jj in 1:n
        rangejj = nzrange(A, jj)
        lenrngjj = length(rangejj)
        for (k, j) in enumerate(rangejj)
            anzj = α * nz[j]
            rvj = rv[j]
            for i in k:lenrngjj
                kk = rangejj[i]
                Cd[rv[kk], rvj] += nz[kk] * anzj
            end
        end
    end
    C
end

rankUpdate!(α::T, A::SparseMatrixCSC{T}, C::HermOrSym{T}) where {T} =
    rankUpdate!(α, A, one(T), C)

rankUpdate!(α::T, A::BlockedSparse{T}, C::HermOrSym{T}) where {T} =
    rankUpdate!(α, A.cscmat, one(T), C)

function rankUpdate!(α::T, A::SparseMatrixCSC{T}, C::Diagonal{T}) where T <: Number
    m, n = size(A)
    dd = C.diag
    @argcheck(length(dd) == m, DimensionMismatch)
    nz = nonzeros(A)
    rv = rowvals(A)
    for j in 1:n
        nzr = nzrange(A, j)
        if !isempty(nzr)
            length(nzr) == 1 || throw(ArgumentError("A*A' has off-diagonal elements"))
            k = nzr[1]
            @inbounds dd[rv[k]] += α * abs2(nz[k])
        end
    end
    C
end

function rankUpdate!(α::T, A::BlockedSparse{T}, C::HermOrSym{T,UniformBlockDiagonal{T}}) where T
    Arb = A.rowblocks
    Cdf = C.data.facevec
    (m = length(Arb)) == length(Cdf) || 
        throw(DimensionMismatch("length(A.rowblocks) = $m ≠ $(length(Cdf)) = length(C.data.facevec)"))
    for (b, d) in zip(Arb, Cdf)
        for v in b
            BLAS.syr!('L', α, v, d)
        end
    end
    C
end
