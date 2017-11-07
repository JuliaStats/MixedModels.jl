function αβA_mul_Bc!(α::T, A::StridedMatrix{T}, B::StridedMatrix{T},
                     β::T, C::StridedMatrix{T}) where T <: BlasFloat
    BLAS.gemm!('N', 'C', α, A, B, β, C)
end

function αβA_mul_Bc!(α::T, A::SparseMatrixCSC{T}, B::SparseMatrixCSC{T},
                     β::T, C::Matrix{T}) where T <: Number
    @argcheck B.m == size(C, 2) && A.m == size(C, 1) && A.n == B.n  DimensionMismatch
    anz = nonzeros(A)
    arv = rowvals(A)
    bnz = nonzeros(B)
    brv = rowvals(B)
    if β ≠ one(T)
        β ≠ zero(T) ? scale!(C, β) : fill!(C, β)
    end
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

function αβA_mul_Bc!(α::T, A::StridedVecOrMat{T}, B::SparseMatrixCSC{T}, β::T,
                     C::StridedVecOrMat{T}) where T
    m, n = size(A)
    p, q = size(B)
    r, s = size(C)
    @argcheck(r == m && s == p && n == q, DimensionMismatch)
    if β ≠ one(T)
        iszero(β) ? fill!(C, β) : scale!(C, β)
    end
    nz = nonzeros(B)
    rv = rowvals(B)
    @inbounds for j in 1:q, k in nzrange(B, j)
        rvk = rv[k]
        anzk = α * nz[k]
        for jj in 1:r  # use .= fusing in v0.6.0 and later
            C[jj, rvk] += A[jj, j] * anzk
        end
    end
    C
end

αβAc_mul_B!(α::T, A::StridedMatrix{T}, B::StridedVector{T}, β::T,
            C::StridedVector{T}) where {T<:BlasFloat} = BLAS.gemv!('C', α, A, B, β, C)

αβAc_mul_B!(α::T, A::SparseMatrixCSC{T}, B::StridedVector{T}, β::T,
            C::StridedVector{T}) where T = Ac_mul_B!(α, A, B, β, C)

function Ac_ldiv_B!(A::LowerTriangular{T,UniformBlockDiagonal{T}}, B::StridedVector{T}) where {T}
    @argcheck length(B) == size(A, 2) DimensionMismatch
    m, n, k = size(A.data.data)
    fv = A.data.facevec
    bb = reshape(B, (n, k))
    for j in 1:k
        Ac_ldiv_B!(LowerTriangular(fv[j]), view(bb, :, j))
    end
    B
end

if VERSION < v"0.7.0-DEV.586"
    Ac_ldiv_B!(D::Diagonal{T}, B::StridedVecOrMat{T}) where {T} = A_ldiv_B!(D, B)

    function A_rdiv_B!(A::StridedMatrix{T}, D::Diagonal{T}) where T
        scale!(A, inv.(D.diag))
        A
    end

    A_rdiv_Bc!(A::StridedMatrix{T}, D::Diagonal{T}) where {T} = A_rdiv_B!(A, D)

    function A_rdiv_Bc!(A::SparseMatrixCSC{T}, D::Diagonal{T}) where T
        if size(D, 2) ≠ size(A, 2)
            throw(DimensionMismatch("size(A,2)=$(size(A,2)) should be size(D, 1)=$(size(D,1))"))
        end
        dd = D.diag
        nonz = nonzeros(A)
        for j in 1:A.n
            ddj = dd[j]
            for k in nzrange(A, j)
                nonz[k] /= ddj
            end
        end
        A
    end
end

function A_rdiv_Bc!(A::Matrix{T}, B::LowerTriangular{T,UniformBlockDiagonal{T}}) where {T}
    m, n, k = size(B.data.data)
    @argcheck size(A, 2) == size(B, 1) && m == n DimensionMismatch
    offset = 0
    one2m = 1:m
    for f in B.data.facevec
        BLAS.trsm!('R', 'L', 'T', 'N', one(T), f, view(A, :, one2m + offset))
        offset += m
    end
    A
end

function A_rdiv_Bc!(A::SparseMatrixCSC{T}, B::LowerTriangular{T,UniformBlockDiagonal{T}}) where {T}
    nz = nonzeros(A)
    offset = 0
    m, n, k = size(B.data.data)
    for f in B.data.facevec
        nzr = nzrange(A, offset + 1).start : nzrange(A, offset + n).stop
        q = div(length(nzr), m)
            ## FIXME Still allocating 1.4 GB.  Call BLAS.trsm directly
        A_rdiv_Bc!(unsafe_wrap(Array, pointer(nz, nzr[1]), (q, m)), LowerTriangular(f))
        offset += n
    end
    A
end
