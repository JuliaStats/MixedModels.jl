function αβA_mul_Bc!(α::T, A::StridedMatrix{T}, B::StridedMatrix{T},
                     β::T, C::StridedMatrix{T}) where T <: BlasFloat
    BLAS.gemm!('N', 'C', α, A, B, β, C)
end

function αβA_mul_Bc!(α::T, A::SparseMatrixCSC{T}, B::SparseMatrixCSC{T},
                     β::T, C::Matrix{T}) where T <: Number
    @argcheck(B.m == size(C, 2) && A.m == size(C, 1) && A.n == B.n, DimensionMismatch)
    anz = nonzeros(A)
    arv = rowvals(A)
    bnz = nonzeros(B)
    brv = rowvals(B)
    β == 1 || rmul!(C, β)
    for j = 1:A.n
        for ib in nzrange(B, j)
            αbnz = α * bnz[ib]
            jj = brv[ib]
            for ia in nzrange(A, j)
                C[arv[ia], jj] += anz[ia] * αbnz
            end
        end
    end
    C
end

αβA_mul_Bc!(α::T, A::BlockedSparse{T}, B::BlockedSparse{T}, β::T, C::Matrix{T}) where {T} =
    αβA_mul_Bc!(α, A.cscmat, B.cscmat, β, C)

function αβA_mul_Bc!(α::T, A::StridedVecOrMat{T}, B::SparseMatrixCSC{T}, β::T,
                     C::StridedVecOrMat{T}) where T
    m, n = size(A)
    p, q = size(B)
    r, s = size(C)
    @argcheck(r == m && s == p && n == q, DimensionMismatch)
    β == 1 || rmul!(C, β)
    nz = nonzeros(B)
    rv = rowvals(B)
    @inbounds for j in 1:q, k in nzrange(B, j)
        rvk = rv[k]
        anzk = α * nz[k]
        for jj in 1:r
            C[jj, rvk] += A[jj, j] * anzk
        end
    end
    C
end

αβA_mul_Bc!(α::T, A::StridedVecOrMat{T}, B::BlockedSparse{T}, β::T,
            C::StridedVecOrMat{T}) where {T} = αβA_mul_Bc!(α, A, B.cscmat, β, C)

αβAc_mul_B!(α::T, A::StridedMatrix{T}, B::StridedVector{T}, β::T,
            C::StridedVector{T}) where {T<:BlasFloat} = BLAS.gemv!('C', α, A, B, β, C)

αβAc_mul_B!(α::T, A::SparseMatrixCSC{T}, B::StridedVector{T}, β::T,
            C::StridedVector{T}) where {T} = mul!(C, adjoint(A), B, α, β)

αβAc_mul_B!(α::T, A::BlockedSparse{T}, B::StridedVector{T}, β::T,
            C::StridedVector{T}) where {T} = αβAc_mul_B!(α, A.cscmat, B, β, C)

function LinearAlgebra.ldiv!(adjA::Adjoint{T,<:LowerTriangular{T,UniformBlockDiagonal{T}}},
                             B::StridedVector{T}) where {T}
    A = adjA.parent
    @argcheck length(B) == size(A, 2) DimensionMismatch
    m, n, k = size(A.data.data)
    fv = A.data.facevec
    bb = reshape(B, (n, k))
    for j in 1:k
        ldiv!(adjoint(LowerTriangular(fv[j])), view(bb, :, j))
    end
    B
end

function LinearAlgebra.rdiv!(A::Matrix{T},
                             adjB::Adjoint{T,<:LowerTriangular{T,UniformBlockDiagonal{T}}}) where T
    Bd = adjB.parent.data
    m, n, k = size(Bd.data)
    @argcheck(size(A, 2) == size(Bd, 1) && m == n, DimensionMismatch)
    inds = 1:m
    for (i, f) in enumerate(Bd.facevec)
        BLAS.trsm!('R', 'L', 'T', 'N', one(T), f, view(A, :, inds .+ m * (i-1)))
    end
    A
end

function LinearAlgebra.rdiv!(A::BlockedSparse{T}, B::Adjoint{T,<:LowerTriangular{T,UniformBlockDiagonal{T}}}) where T
    Bp = B.parent
    @argcheck(length(A.colblocks) == length(Bp.data.facevec), DimensionMismatch)
    for (b,f) in zip(A.colblocks, Bp.data.facevec)
        rdiv!(b, adjoint(LowerTriangular(f)))
    end
    A
end
