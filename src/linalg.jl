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
    Adat = A.data.data
    m, n, k = size(Adat)
    bb = reshape(B, (n, k))
    for j in axes(Adat, 3)
        ldiv!(adjoint(LowerTriangular(view(Adat, :, :, j))), view(bb, :, j))
    end
    B
end

function LinearAlgebra.rdiv!(
    A::Matrix{T},
    adjB::Adjoint{T,<:LowerTriangular{T,UniformBlockDiagonal{T}}},
) where {T}
    m, n = size(A)
    Bd = adjB.parent.data
    Bdd = Bd.data
    r, s, blk = size(Bdd)
    n == size(Bd, 1) && r == s || throw(DimensionMismatch())
    for b = axes(Bd.data, 3)
        coloffset = (b - 1) * s
        rdiv!(view(A, :, coloffset+1:coloffset+s), adjoint(LowerTriangular(view(Bdd, :, :, b))))
    end
    A
end

function LinearAlgebra.rdiv!(
    A::BlockedSparse{T,S,P},
    B::Adjoint{T,<:LowerTriangular{T,UniformBlockDiagonal{T}}},
) where {T,S,P}
    Bpd = B.parent.data
    Bdat = Bpd.data
    j, k, l = size(Bdat)
    cbpt = A.colblkptr
    nzv = A.cscmat.nzval
    P == j == k && length(cbpt) == (l + 1) || throw(DimensionMismatch(""))
    for j in axes(Bdat, 3)
        rdiv!(
            reshape(view(nzv,cbpt[j]:(cbpt[j+1]-1)),:,P),
            adjoint(LowerTriangular(view(Bdat,:,:,j)))
            )
    end
    A
end
