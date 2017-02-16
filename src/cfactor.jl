function cholBlocked!{T}(A::Matrix{AbstractMatrix{T}}, ::Type{Val{:L}})
    n, A11 = LinAlg.checksquare(A), A[1, 1]
    cholUnblocked!(A11, Val{:L})
    A11lt = isa(A11, Diagonal) ? A11 : LowerTriangular(A11)
    for i in 2:n
        LinAlg.A_rdiv_Bc!(A[i, 1], A11lt)
    end
    for i in 2:n
        Ai1 = A[i, 1]
        rankUpdate!(-one(T), Ai1, A[i, i])
        for j in (i + 1):n
            rankUpdate!(-one(T), Ai1, A[j, 1], A[i, j])
        end
    end
    if n > 1
        cholBlocked!(view(A, 2:n, 2:n))
    end
    return A
end

function cholUnblocked!{T <: AbstractFloat}(D::Diagonal{T}, ::Type{Val{:L}})
    map!(sqrt, D.diag)
    D
end

A_mul_Bc!{T<:BlasFloat}(α::T, A::StridedMatrix{T}, B::StridedMatrix{T}, β::T, C::StridedMatrix{T}) = Base.BLAS.gemm!('N', 'C', α, A, B, β, C)

cholUnblocked!{T <: AbstractFloat}(D::Diagonal{T}, ::Type{Val{:U}}) = cholUnblocked!(D, Val{:L})

function cholBlocked!{T}(A::HBlkDiag{T}, ::Type{Val{:L}})
    Aa = A.arr
    r, s, t = size(Aa)
    if r ≠ s
        throw(ArgumentError("HBlkDiag matrix A must be square"))
    end
    scm = Array(T, (r, r))
    for k in 1 : t  # FIXME: Lots of allocations in this loop
        for j in 1 : r, i in 1 : j
            scm[i, j] = Aa[i, j, k]
        end
        LAPACK.potrf!('L', scm)
        for j in 1 : r, i in 1 : j
            Aa[i, j, k] = scm[i, j]
        end
    end
    UpperTriangular(A)
end

function rankUpdate!{T <: Number}(α::T, A::SparseMatrixCSC{T}, C::Hermitian{T, Diagonal{T}})
    m, n = size(A)
    dd = C.data.diag
    if length(dd) ≠ m
        throw(DimensionMismatch("size(C,2) = $(length(dd)) ≠ $m = size(A,1)"))
    end
    nz = nonzeros(A)
    rv = rowvals(A)
    for j in 1 : n
        nzr = nzrange(A, j)
        length(nzr) == 1 || throw(ArgumentError("A*A' has off-diagonal elements"))
        k = nzr[1]
        @inbounds dd[rv[k]] += α * abs2(nz[k])
    end
    return C
end

## Probably don't need this method.  Created because I misread an error message.
function Base.LinAlg.A_mul_Bc!{T<:Number}(α::T, A::SparseMatrixCSC{T}, B::StridedVecOrMat{T},
    β::T, C::StridedVecOrMat{T})
    size(B, 1) == size(C, 2) || throw(DimensionMismatch())
    nzv = A.nzval
    rv = A.rowval
    if β != 1
        β != 0 ? scale!(C, β) : fill!(C, zero(eltype(C)))
    end
    for col = 1:A.n
        for k = 1:size(C, 2)
            αxj = α*B[k, col]
            @inbounds for j = nzrange(A, col)
                C[rv[j], k] += nzv[j]*αxj
            end
        end
    end
    C
end

function Base.LinAlg.A_mul_Bc!{T<:Number}(α::T, A::StridedVecOrMat{T}, B::SparseMatrixCSC{T},
    β::T, C::StridedVecOrMat{T})
    (m, n), (p, q), (r, s) = size(A), size(B), size(C)
    if r ≠ m || s ≠ p || n ≠ q
        throw(DimensionMismatch("size(C,1) ≠ size(A,1) or size(C,2) ≠ size(B,1) or size(A,2) ≠ size(B,2)"))
    end
    if β != 1
        β != 0 ? scale!(C, β) : fill!(C, zero(eltype(C)))
    end
    nz, rv = nonzeros(B), rowvals(B)
    for j in 1 : q, k in nzrange(B, j)
        rvk = rv[k]
        anzk = α * nz[k]
        for jj in 1 : r  # use .= fusing in v0.6.0 and later
            C[jj, rvk] += A[jj, j] * anzk
        end
    end
    return C
end

"""
    downdate!(C::AbstractMatrix, A::AbstractMatrix)
    downdate!(C::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix)

Subtracts, in place, `A * A'` or `A * B'` from `C`
"""
function downdate! end

downdate!{T <: LinAlg.BlasFloat}(C::DenseMatrix{T}, A::DenseMatrix{T}) =
    BLAS.syrk!('L', 'N', -one(T), A, one(T), C)

downdate!{T <: LinAlg.BlasFloat}(C::DenseMatrix{T}, A::DenseMatrix{T}, B::DenseMatrix{T}) =
    BLAS.gemm!('N', 'T', -one(T), A, B, one(T), C)

function downdate!{T}(C::Diagonal{T}, A::SparseMatrixCSC{T})
    m, n = size(A)
    dd = C.diag
    if length(dd) ≠ m
        throw(DimensionMismatch("size(C,2) = $(length(dd)) ≠ $m = size(A,1)"))
    end
    nz = nonzeros(A)
    rv = rowvals(A)
    for j in 1 : n
        for k in nzrange(A, j)
            @inbounds dd[rv[k]] -= abs2(nz[k])
        end
    end
    C
end

function downdate!{T}(C::Diagonal{T},A::Diagonal{T})
    if size(C) ≠ size(A)
        throw(DimensionMismatch("size(C) ≠ size(A)"))
    end
    map!((c, a) -> c - abs2(a), C.diag, C.diag, A.diag)
    C
end

function downdate!{T}(C::Diagonal{T}, A::Diagonal{T}, B::Diagonal{T})
    if !(size(C) == size(A) == size(B))
        throw(DimensionMismatch("need size(C) == size(A) == size(B)"))
    end
    map!((c, a, b) -> c - a * b, C.diag, C.diag, A.diag, B.diag)
    C
function downdate!{T}(C::DenseMatrix{T}, A::Diagonal{T}, B::DenseMatrix{T})
    a = A.diag
    if ((m, n) = size(B)) ≠ size(C)
        throw(DimensionMismatch("size(B) ≠ size(C)"))
    end
    if length(a) ≠ m
        throw(DimensionMismatch("size(A,2) ≠ size(B,1)"))
    end
    for j in 1:n, i in 1:m
        C[i,j] -= a[i] * B[i,j]
    end
    C
end

function downdate!{T}(C::DenseMatrix{T}, A::SparseMatrixCSC{T}, B::DenseMatrix{T})
    (m, n), (p, q), (r, s) = size(A), size(B), size(C)
    if r ≠ n || s ≠ q || m ≠ p
        throw(DimensionMismatch("size(C,1) ≠ size(A,2) or size(C,2) ≠ size(B,2) or size(A,1) ≠ size(B,1)"))
    end
    nz = nonzeros(A)
    rv = rowvals(A)
    for j in 1 : n, k in nzrange(A,j)
        rvk = rv[k]
        for jj in 1 : s
            C[rvk, jj] -= nz[k] * B[rvk, jj]
        end
    end
    C
end
function downdate!{T}(C::DenseMatrix{T}, A::DenseMatrix{T}, B::SparseMatrixCSC{T})
    (m, n), (p, q), (r, s) = size(A), size(B), size(C)
    if r ≠ m || s ≠ p || n ≠ q
        throw(DimensionMismatch("size(C,1) ≠ size(A,1) or size(C,2) ≠ size(B,1) or size(A,2) ≠ size(B,2)"))
    end
    nz, rv = nonzeros(B), rowvals(B)
    for j in 1 : q, k in nzrange(B, j)
        rvk = rv[k]
        nzk = nz[k]
        for jj in 1 : r  # use .= fusing in v0.6.0 and later
            C[jj, rvk] -= A[jj, j] * nzk
        end
    end
    C
end

function downdate!{T}(C::SparseMatrixCSC{T}, A::SparseMatrixCSC{T}, B::SparseMatrixCSC{T})
    AtB = A'B
    ((m, n) = size(C)) == size(AtB) || throw(DimensionMismatch("size(C) ≠ size(A'B)"))
    atbrv, atbnz, crv, cnz = rowvals(AtB), nonzeros(AtB), rowvals(C), nonzeros(C)
    for j in 1 : n
        nzrCj = nzrange(C, j)
        rowsCj = crv[nzrCj]
        lenrowsCj = length(rowsCj)
        for k in nzrange(AtB, j)
            i = atbrv[k]
            ck = searchsortedfirst(rowsCj, i)
            if ck > lenrowsCj || rowsCj[ck] ≠ i
                throw(ArgumentError(string("nonzero rows in $j'th column of A'B are not",
                    " a subset of those in C")))
            end
            cnz[nzrCj[ck]] -= atbnz[k]
        end
    end
    C
end

function downdate!{T}(C::DenseMatrix{T}, A::SparseMatrixCSC{T})
    m, n = size(A)
    if m ≠ LinAlg.checksquare(C)
        throw(DimensionMismatch("C is not square or size(C, 2) ≠ size(A, 1)"))
    end
    rv, nz = rowvals(A), nonzeros(A)
    for jj in 1 : n
        rangejj = nzrange(A, jj)
        for j in rangejj
            nzj, rvj = nz[j], rv[j]
            for i in rangejj
                C[rv[i], rvj] -= nz[i] * nzj
            end
        end
    end
    C
end

## FIXME: Need a downdate! method for SparseMatrixCSC, SparseMatrixCSC, SparseMatrixCSC (3 or more nested)

"""
    inflate!(A::HblkDiag)
    inflate!(A::Diagonal)
    inflate!(A::StridedMatrix)

Adds an identity to `A` in place.
"""
function inflate! end

function inflate!(A::HBlkDiag)  # change to Diagonal{Mmatrix}
    Aa = A.arr
    r, s, k = size(Aa)
    for j in 1 : k, i in 1 : min(r, s)
        Aa[i, i, j] += 1
    end
    A
end
inflate!{T <: AbstractFloat}(A::StridedMatrix{T}) = (A += I)
inflate!{T <: AbstractFloat}(D::Diagonal{T}) = (D += I)
