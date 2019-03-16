function mulαβ!(C::Matrix{T}, A::Matrix{T}, adjB::Adjoint{T,<:Matrix{T}}, 
        α=true, β=false) where {T<:BlasFloat}
    BLAS.gemm!('N', 'C', T(α), A, adjB.parent, T(β), C)
end

function mulαβ!(C::Matrix{T}, A::SparseMatrixCSC{T}, adjB::Adjoint{T,<:SparseMatrixCSC{T}},
        α=true, β=false) where T <: Number
    B = adjB.parent
    B.m == size(C, 2) && A.m == size(C, 1) && A.n == B.n || throw(DimensionMismatch(""))
    anz = nonzeros(A)
    arv = rowvals(A)
    bnz = nonzeros(B)
    brv = rowvals(B)
    isone(β) || rmul!(C, β)
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

mulαβ!(C::Matrix{T}, A::BlockedSparse{T}, adjB::Adjoint{T,<:BlockedSparse{T}}, α=true,
       β=false) where {T} = mulαβ!(C, A.cscmat, adjB.parent.cscmat', α, β)

function mulαβ!(C::Matrix{T}, A::SparseMatrixCSC{T}, adjB::Adjoint{T,Matrix{T}},
        α=true, β=false) where {T}
    B = adjB.parent
    A.n == size(B, 2) || throw(DimensionMismatch())
    A.m == size(C, 1) || throw(DimensionMismatch())
    size(B, 1) == size(C, 2) || throw(DimensionMismatch())
    nzv = A.nzval
    rv = A.rowval
    if β != 1
        β != 0 ? rmul!(C, β) : fill!(C, zero(eltype(C)))
    end
    for k = 1:size(C, 2)
        @inbounds for col = 1:A.n
            αxj = α*B[k,col]
            for j = nzrange(A, col)
                C[rv[j], k] += nzv[j]*αxj
            end
        end
    end
    C
end

mulαβ!(C::Matrix{T}, A::BlockedSparse{T}, adjB::Adjoint{T,<:Matrix{T}}, α=true, β=false) where {T} =
    mulαβ!(C, A.cscmat, adjB, α, β)

function mulαβ!(C::SparseMatrixCSC{T}, A::SparseMatrixCSC{T}, adjB::Adjoint{T,<:SparseMatrixCSC{T}},
        α=true, β=false) where {T}
    B = adjB.parent
    C.m == A.m && C.n == B.m && A.n == B.n || throw(DimensionMismatch(""))
    Anz = nonzeros(A)
    Bnz = nonzeros(B)
    Cnz = nonzeros(C)
    isone(β) || rmul!(Cnz, β)
    Arv = rowvals(A)
    Brv = rowvals(B)
    Crv = rowvals(C)
    for j in 1:A.n
        for K in nzrange(B, j)
            k = Brv[K]
            alphabjk = α * Bnz[K]
            colkfirstr = Int(C.colptr[k])
            colklastr = Int(C.colptr[k + 1] - 1)
            for I in nzrange(A, j)
                i = Arv[I]
                searchk = searchsortedfirst(Crv, i, colkfirstr, colklastr, Base.Order.Forward)
                if searchk <= colklastr && Crv[searchk] == i
                    Cnz[searchk] += alphabjk * Anz[I]
                else
                    throw(ArgumentError("C does not have the nonzero pattern of A*B'"))
                end
            end
        end
    end
    C
end

mulαβ!(C::BlockedSparse{T}, A::BlockedSparse{T}, adjB::Adjoint{T,<:BlockedSparse{T}}, α=true, β=false) where {T} =
    mulαβ!(C.cscmat, A.cscmat, adjB.parent.cscmat', α, β)

function mulαβ!(C::StridedVecOrMat{T}, A::StridedVecOrMat{T}, adjB::Adjoint{T,<:SparseMatrixCSC{T}},
        α=true, β=false) where T
    B = adjB.parent
    m, n = size(A)
    p, q = size(B)
    r, s = size(C)
    r == m && s == p && n == q || throw(DimensionMismatch(""))
    isone(β) || rmul!(C, β)
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

mulαβ!(C::StridedVecOrMat{T}, A::StridedVecOrMat{T}, adjB::Adjoint{T,<:BlockedSparse{T}},
    α=true, β=false) where {T} = mulαβ!(C, A, adjB.parent.cscmat', α, β)

mulαβ!(C::StridedVector{T}, adjA::Adjoint{T,<:StridedMatrix{T}}, B::StridedVector{T},
    α=true, β=false) where {T<:BlasFloat} = BLAS.gemv!('C', T(α), adjA.parent, B, T(β), C)

mulαβ!( C::StridedVector{T}, adjA::Adjoint{T,<:SparseMatrixCSC{T}}, B::StridedVector{T},
    α=true, β=false) where {T} = mul!(C, adjA, B, T(α), T(β))

mulαβ!(C::StridedVector{T}, adjA::Adjoint{T,<:BlockedSparse{T}}, B::StridedVector{T},
    α=true, β=false) where {T} = mulαβ!(α, adjA.parent.cscmat', B, β, C)

function LinearAlgebra.ldiv!(adjA::Adjoint{T,<:LowerTriangular{T,UniformBlockDiagonal{T}}},
        B::StridedVector{T}) where {T}
    A = adjA.parent
    length(B) == size(A, 2) || throw(DimensionMismatch(""))
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
    size(A, 2) == size(Bd, 1) && m == n || throw(DimensionMismatch(""))
    inds = 1:m
    for (i, f) in enumerate(Bd.facevec)
        BLAS.trsm!('R', 'L', 'T', 'N', one(T), f, view(A, :, inds .+ m * (i-1)))
    end
    A
end

function LinearAlgebra.rdiv!(A::SparseMatrixCSC{T},
        adjB::Adjoint{T,<:LowerTriangular{T,<:Diagonal{T}}}) where T
    rdiv!(A, adjB.parent.data)
end

function LinearAlgebra.rdiv!(A::BlockedSparse{T},
        B::Adjoint{T,<:LowerTriangular{T,UniformBlockDiagonal{T}}}) where T
    Bp = B.parent
    length(A.colblocks) == length(Bp.data.facevec) || throw(DimensionMismatch(""))
    for (b,f) in zip(A.colblocks, Bp.data.facevec)
        rdiv!(b, adjoint(LowerTriangular(f)))
    end
    A
end

if VERSION <= v"1.1.0-DEV.421"
    LinearAlgebra.mul!(C::AbstractMatrix, A::AbstractMatrix, J::UniformScaling) = mul!(C, A, J.λ)
    LinearAlgebra.mul!(C::AbstractVecOrMat, J::UniformScaling, B::AbstractVecOrMat) = mul!(C, J.λ, B)
    LinearAlgebra.rmul!(A::AbstractMatrix, J::UniformScaling) = rmul!(A, J.λ)
    LinearAlgebra.lmul!(J::UniformScaling, B::AbstractVecOrMat) = lmul!(J.λ, B)
end
