"""
    rankUpdate!(A, C)
    rankUpdate!(α, A, C)
    rankUpdate!(α, A, β, C)

A rank-k update of a Hermitian (Symmetric) matrix.

`α` and `β` both default to 1.0.  When `α` is -1.0 this is a downdate operation.
The name `rankUpdate!` is borrowed from [https://github.com/andreasnoack/LinearAlgebra.jl]
"""
function rankUpdate! end

rankUpdate!{T<:BlasReal,S<:StridedMatrix}(α::T, a::StridedVector{T}, A::HermOrSym{T,S}) = BLAS.syr!(A.uplo, α, a, A.data)
rankUpdate!{T<:BlasReal,S<:StridedMatrix}(a::StridedVector{T}, A::HermOrSym{T,S}) = rankUpdate!(one(T), a, A)

rankUpdate!{T<:BlasReal,S<:StridedMatrix}(α::T, A::StridedMatrix{T}, β::T, C::HermOrSym{T,S}) = BLAS.syrk!(C.uplo, 'N', α, A, β, C.data)
rankUpdate!{T<:Real,S<:StridedMatrix}(α::T, A::StridedMatrix{T}, C::HermOrSym{T,S}) = rankUpdate!(α, A, one(T), C)
rankUpdate!{T<:Real,S<:StridedMatrix}(A::StridedMatrix{T}, C::HermOrSym{T,S}) = rankUpdate!(one(T), A, one(T), C)

function rankUpdate!{T<:AbstractFloat,S<:StridedMatrix}(α::T, A::SparseMatrixCSC{T}, β::T, C::HermOrSym{T,S})
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

rankUpdate!{T<:AbstractFloat,S<:StridedMatrix}(α::T, A::SparseMatrixCSC{T}, C::HermOrSym{T,S}) = rankUpdate!(α, A, one(T), C)

function rankUpdate!{T <: Number}(α::T, A::SparseMatrixCSC{T}, C::Diagonal{T})
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

function rankUpdate!{T<:Number}(α::T, A::SparseMatrixCSC{T}, C::Diagonal{LowerTriangular{T,Matrix{T}}})
    m, n = size(A)
    cdiag = C.diag
    dsize = size.(cdiag, 2)
    @argcheck sum(dsize) == m DimensionMismatch
    if all(dsize .== 1)
        nz = nonzeros(A)
        rv = rowvals(A)
        for j in 1:n
            nzr = nzrange(A, j)
            length(nzr) == 1 || throw(ArgumentError("A*A' has off-diagonal elements"))
            k = nzr[1]
            @inbounds cdiag[rv[k]].data[1] += α * abs2(nz[k])
        end
    else  # not efficient but only used for nested vector-valued r.e.'s, which are rare
        aat = α * (A * A')
        nz = nonzeros(aat)
        rv = rowvals(aat)
        offset = 0
        for d in cdiag
            k = size(d, 2)
            for j in 1:k
                for i in nzrange(aat, offset + j)
                    ii = rv[i] - offset
                    0 < ii ≤ k || throw(ArgumentError("A*A' does not conform to B"))
                    if ii ≥ j  # update lower triangle only
                        d.data[ii, j] += nz[i]
                    end
                end
            end
            offset += k
        end
    end
    C
end
