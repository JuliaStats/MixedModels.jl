function LinearAlgebra.mul!(
    C::Matrix{T},
    blkA::BlockedSparse{T},
    adjB::Adjoint{T,<:BlockedSparse{T}},
    α::Number,
    β::Number,
) where {T}
    A = blkA.cscmat
    B = adjB.parent.cscmat
    B.m == size(C, 2) && A.m == size(C, 1) && A.n == B.n || throw(DimensionMismatch(""))
    anz = nonzeros(A)
    arv = rowvals(A)
    bnz = nonzeros(B)
    brv = rowvals(B)
    isone(β) || rmul!(C, β)
    @inbounds for j = 1:A.n
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

LinearAlgebra.mul!(
    C::Matrix{T},
    A::BlockedSparse{T},
    adjB::Adjoint{T,<:Matrix{T}},
    α::Number,
    β::Number,
) where {T} = mul!(C, A.cscmat, adjB, α, β)

LinearAlgebra.mul!(
    C::BlockedSparse{T},
    A::BlockedSparse{T},
    adjB::Adjoint{T,<:BlockedSparse{T}},
    α::Number,
    β::Number,
) where {T} = mul!(C.cscmat, A.cscmat, adjB.parent.cscmat', α, β)

LinearAlgebra.mul!(
    C::StridedVecOrMat{T},
    A::StridedVecOrMat{T},
    adjB::Adjoint{T,<:BlockedSparse{T}},
    α::Number,
    β::Number,
) where {T} = mul!(C, A, adjB.parent.cscmat', α, β)

LinearAlgebra.mul!(
    C::StridedVector{T},
    adjA::Adjoint{T,<:BlockedSparse{T}},
    B::StridedVector{T},
    α::Number,
    β::Number,
) where {T} = mul!(C, adjA.parent.cscmat', B, α, β)

function LinearAlgebra.ldiv!(
    adjA::Adjoint{T,<:LowerTriangular{T,UniformBlockDiagonal{T}}},
    B::StridedVector{T},
) where {T}
    A = adjA.parent
    length(B) == size(A, 2) || throw(DimensionMismatch(""))
    m, n, k = size(A.data.data)
    fv = A.data.facevec
    bb = reshape(B, (n, k))
    for j = 1:k
        ldiv!(adjoint(LowerTriangular(fv[j])), view(bb, :, j))
    end
    B
end

function LinearAlgebra.rdiv!(
    A::Matrix{T},
    adjB::Adjoint{T,<:LowerTriangular{T,UniformBlockDiagonal{T}}},
) where {T}
    m, n = size(A)
    Bd = adjB.parent.data
    r, s, blk = size(Bd.data)
    n == size(Bd, 1) && r == s || throw(DimensionMismatch())
    @inbounds for b = 1:blk
        coloffset = (b - 1) * s
        for i = 1:m
            for j = 1:s
                Aij = A[i, j+coloffset]
                for k = 1:j-1
                    Aij -= A[i, k+coloffset] * Bd.data[j, k, b]'
                end
                A[i, j+coloffset] = Aij / Bd.data[j, j, b]'
            end
        end
    end
    A
end

function LinearAlgebra.rdiv!(
    A::BlockedSparse{T,S,P},
    B::Adjoint{T,<:LowerTriangular{T,UniformBlockDiagonal{T}}},
) where {T,S,P}
    Bpd = B.parent.data
    j, k, l = size(Bpd.data)
    cbpt = A.colblkptr
    nzv = A.cscmat.nzval
    P == j == k && length(cbpt) == (l + 1) || throw(DimensionMismatch(""))
    for (j, f) in enumerate(Bpd.facevec)
        rdiv!(reshape(view(nzv, cbpt[j]:(cbpt[j+1]-1)), :, P), adjoint(LowerTriangular(f)))
    end
    A
end
