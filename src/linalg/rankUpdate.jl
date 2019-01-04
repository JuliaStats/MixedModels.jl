"""
    rankUpdate!(C, A)
    rankUpdate!(C, A, α)
    rankUpdate!(C, A, α, β)

A rank-k update, C := β*C + α*A'A, of a Hermitian (Symmetric) matrix.

`α` and `β` both default to 1.0.  When `α` is -1.0 this is a downdate operation.
The name `rankUpdate!` is borrowed from [https://github.com/andreasnoack/LinearAlgebra.jl]
The order of the arguments
"""
function rankUpdate! end

function rankUpdate!(C::HermOrSym{T,S}, a::StridedVector{T},  
        α=true) where {T<:BlasReal,S<:StridedMatrix}
    BLAS.syr!(C.uplo, T(α), a, C.data)
    C  ## to ensure that the return value is HermOrSym
end

function rankUpdate!(C::HermOrSym{T,S}, A::StridedMatrix{T},
        α=true, β=true) where {T<:BlasReal,S<:StridedMatrix}
    BLAS.syrk!(C.uplo, 'N', T(α), A, T(β), C.data)
    C
end

function rankUpdate!(C::HermOrSym{T,S}, A::SparseMatrixCSC{T}, α=true, β=true) where {T,S}
    m, n = size(A)
    @argcheck(m == size(C, 2), DimensionMismatch)
    @argcheck(C.uplo == 'L', ArgumentError)
    Cd = C.data
    isone(β) || rmul!(LowerTriangular(Cd), β)
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

function rankUpdate!(C::Diagonal{T}, A::SparseMatrixCSC{T}, α=true, β=true) where {T <: Number}
    m, n = size(A)
    dd = C.diag
    @argcheck(length(dd) == m, DimensionMismatch)
    isone(β) || rmul!(dd, β)
    nz = nonzeros(A)
    rv = rowvals(A)
    @inbounds for j in 1:n
        nzr = nzrange(A, j)
        if !isempty(nzr)
            length(nzr) == 1 || throw(ArgumentError("A*A' has off-diagonal elements"))
            k = nzr[1]
            dd[rv[k]] += α * abs2(nz[k])
        end
    end
    C
end

function rankUpdate!(C::HermOrSym{T,UniformBlockDiagonal{T}}, A::BlockedSparse{T},
        α=true) where {T}
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

rankUpdate!(C::HermOrSym{T,Matrix{T}}, A::BlockedSparse{T}, α=true) where {T} = rankUpdate!(C, A.cscmat, α)
