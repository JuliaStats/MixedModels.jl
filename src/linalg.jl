function A_mul_Bc!{T<:BlasFloat}(α::T, A::StridedMatrix{T}, B::StridedMatrix{T},
    β::T, C::StridedMatrix{T})
    BLAS.gemm!('N', 'C', α, A, B, β, C)
end

function A_mul_Bc!{T<:Number}(α::T, A::SparseMatrixCSC{T}, B::SparseMatrixCSC{T},
    β::T, C::Matrix{T})
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

function A_mul_Bc!{T<:Number}(α::T, A::SparseMatrixCSC{T}, B::SparseMatrixCSC{T},
    β::T, C::SparseMatrixCSC{T})
    @argcheck B.m == C.n && A.m == C.m && A.n == B.n  DimensionMismatch
    anz = nonzeros(A)
    arv = rowvals(A)
    bnz = nonzeros(B)
    brv = rowvals(B)
    cnz = nonzeros(C)
    crv = rowvals(C)
    if β ≠ one(T)
        iszero(β) ? fill!(cnz, β) : scale!(cnz, β)
    end
    for j = 1:A.n
        for ib in nzrange(B, j)
            αbnz = α * bnz[ib]
            jj = brv[ib]
            for ia in nzrange(A, j)
                crng = nzrange(C, jj)
                ind = findfirst(crv[crng], arv[ia])
                if iszero(ind)
                    throw(ArgumentError("A*B' has nonzero positions not in C"))
                end
                cnz[crng[ind]] += anz[ia] * αbnz
            end
        end
    end
    C
end

function A_mul_Bc!{T<:Number}(α::T, A::StridedVecOrMat{T}, B::SparseMatrixCSC{T},
    β::T, C::StridedVecOrMat{T})
    m, n = size(A)
    p, q = size(B)
    r, s = size(C)
    if r ≠ m || s ≠ p || n ≠ q
        throw(DimensionMismatch("size(C,1) ≠ size(A,1) or size(C,2) ≠ size(B,1) or size(A,2) ≠ size(B,2)"))
    end
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

Ac_mul_B!{T<:BlasFloat}(α::T, A::StridedMatrix{T}, B::StridedVector{T}, β::T,
    C::StridedVector{T}) = BLAS.gemv!('C', α, A, B, β, C)

function Ac_ldiv_B!{T<:AbstractFloat}(A::Diagonal{LowerTriangular{T,Matrix{T}}}, B::StridedVector{T})
    offset = 0
    for a in A.diag
        k = size(a, 1)
        Ac_ldiv_B!(a, view(B, (1:k) + offset))
        offset += k
    end
    B
end

Ac_ldiv_B!{T}(D::Diagonal{T}, B::StridedVecOrMat{T}) = A_ldiv_B!(D, B)

if false
function A_ldiv_B!{T}(D::Diagonal{T}, B::Diagonal{T})
    @argcheck size(D) == size(B) DimensionMismatch
    map!(/, B.diag, B.diag, D.diag)
    B
end

function A_ldiv_B!{T}(D::Diagonal{T}, B::SparseMatrixCSC{T})
    @argcheck size(D, 2) == size(B, 1) DimensionMismatch
    dd = D.diag
    vals = nonzeros(B)
    rows = rowvals(B)
    @inbounds for k in eachindex(vals)
        vals[k] /= dd[rows[k]]
    end
    B
end
end

function A_rdiv_B!{T}(A::StridedMatrix{T}, D::Diagonal{T})
    scale!(A, inv.(D.diag))
    A
end

if false
function A_rdiv_B!{T}(A::StridedMatrix{T}, D::Diagonal{LowerTriangular{T, Matrix{T}}})
    offset = 0
    for L in D.diag
        k = size(L, 1)
        A_rdiv_B!(view(A, :, (1:k) + offset), L)
        offset += k
    end
    A
end
end

A_rdiv_Bc!{T}(A::StridedMatrix{T}, D::Diagonal{T}) = LinAlg.A_rdiv_B!(A, D)

function A_rdiv_Bc!{T}(A::SparseMatrixCSC{T}, D::Diagonal{T})
    @argcheck size(D, 2) == size(A, 2) DimensionMismatch
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

function A_rdiv_Bc!{T<:AbstractFloat}(A::Matrix, B::Diagonal{LowerTriangular{T,Matrix{T}}})
    offset = 0
    for d in B.diag
        k = size(d, 1)
        ## FIXME all BLAS.trsm directly
        A_rdiv_Bc!(view(A, :, (1:k) + offset), d)
        offset += k
    end
    A
end

function A_rdiv_Bc!{T}(A::SparseMatrixCSC{T}, B::Diagonal{LowerTriangular{T,Matrix{T}}})
    nz = nonzeros(A)
    offset = 0
    for d in B.diag
        if (k = size(d, 1)) == 1
            d1 = d[1]
            offset += 1
            for k in nzrange(A, offset)
                nz[k] /= d1
            end
        else
            nzr = nzrange(A, offset + 1).start : nzrange(A, offset + k).stop
            q = div(length(nzr), k)
            ## FIXME Still allocating 1.4 GB.  Call BLAS.trsm directly
            A_rdiv_Bc!(unsafe_wrap(Array, pointer(nz, nzr[1]), (q, k)), d)
            offset += k
        end
    end
    A
end

function full{T}(A::Diagonal{LowerTriangular{T,Matrix{T}}})
    D = diag(A)
    sz = size.(D, 2)
    n = sum(sz)
    B = zeros(n, n)
    offset = 0
    for (d,s) in zip(D, sz)
        for j in 1:s, i in j:s
            B[offset + i, offset + j] = d[i,j]
        end
        offset += s
    end
    B
end

function rowlengths{T}(A::FactorReTerm{T})
    ld = A.Λ
    [norm(view(ld, i, 1:i)) for i in 1:size(ld, 1)]
end

rowlengths{T}(A::MatrixTerm{T}) = T[]
