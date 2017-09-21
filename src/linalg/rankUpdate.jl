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
rankUpdate!(a::StridedVector{T}, A::HermOrSym{T,S}) where {T<:BlasReal,S<:StridedMatrix} = rankUpdate!(one(T), a, A)

rankUpdate!(α::T, A::StridedMatrix{T}, β::T,
C::HermOrSym{T,S}) where {T<:BlasReal,S<:StridedMatrix} = BLAS.syrk!(C.uplo, 'N', α, A, β, C.data)
rankUpdate!(α::T, A::StridedMatrix{T}, C::HermOrSym{T,S}) where {T<:Real,S<:StridedMatrix} =
    rankUpdate!(α, A, one(T), C)
rankUpdate!(A::StridedMatrix{T}, C::HermOrSym{T,S}) where {T<:Real,S<:StridedMatrix} =
    rankUpdate!(one(T), A, one(T), C)

function rankUpdate!(α::T, A::SparseMatrixCSC{T,I},
                     β::T, C::HermOrSym{T,S}) where {T,I,S<:StridedMatrix{T}}
    m, n = size(A)
    @argcheck m == size(C, 2) && C.uplo == 'L' DimensionMismatch
    Cd = C.data
    if β ≠ one(T)
        scale!(LowerTriangular(Cd), β)
    end
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

function rankUpdate!(α::T, A::SparseMatrixCSC{T}, C::Diagonal{T}) where T <: Number
    m, n = size(A)
    dd = C.diag
    @argcheck length(dd) == m DimensionMismatch
    nz = nonzeros(A)
    rv = rowvals(A)
    for j in 1:n
        nzr = nzrange(A, j)
        length(nzr) == 1 || throw(ArgumentError("A*A' has off-diagonal elements"))
        k = nzr[1]
        @inbounds dd[rv[k]] += α * abs2(nz[k])
    end
    C
end

function rankUpdate!(α::T, A::SparseMatrixCSC{T},
                     C::HermOrSym{T,UniformBlockDiagonal{T}}) where T<:Number
    m, n, k = size(C.data.data)
    @argcheck m == n && size(A, 1) == m * k DimensionMismatch
    aat = α * (A * A')
    nz = nonzeros(aat)
    rv = rowvals(aat)
    offset = 0
    for f in C.data.facevec
        for j in 1:m
            for i in nzrange(aat, offset + j)
                ii = rv[i] - offset
                0 < ii ≤ k || throw(ArgumentError("A*A' does not conform to B"))
                if ii ≥ j  # update lower triangle only
                    f[ii, j] += nz[i]
                end
            end
        end
        offset += m
    end
    C
end
