"""
    cfactor!(A::AbstractMatrix)

A slightly modified version of `chol!` from `Base`

Uses `inject!` (as opposed to `copy!`), `downdate!` (as opposed to `syrk!` or `gemm!`)
and recursive calls to `cfactor!`.

Note: The `cfactor!` method for dense matrices calls `LAPACK.potrf!` directly to avoid
errors being thrown when `A` is computationally singular
"""
function cfactor!(A::AbstractMatrix)
    n = LinAlg.checksquare(A)
    for k = 1:n
        Akk = A[k, k]
        for i in 1:(k - 1)
            downdate!(Akk, A[i, k])  # A[k,k] -= A[i,k]'A[i,k]
        end
        Akk = cfactor!(Akk)          # right Cholesky factor of A[k,k]
        for j in (k + 1):n
            for i in 1:(k - 1)
                downdate!(A[k, j], A[i, k], A[i, j]) # A[k,j] -= A[i,k]'A[i,j]
            end
            LinAlg.Ac_ldiv_B!(Akk, A[k, j])
        end
    end
    UpperTriangular(A)
end

function cfactor!{T <: AbstractFloat}(D::Diagonal{T})
    dd = D.diag
    for i in eachindex(dd)
        dd[i] = sqrt(dd[i])
    end
    D
end

cfactor!{T <: LinAlg.BlasFloat}(A::Matrix{T}) = UpperTriangular(LAPACK.potrf!('U', A)[1])

function cfactor!{T}(A::HBlkDiag{T})
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
        LAPACK.potrf!('U', scm)
        for j in 1 : r, i in 1 : j
            Aa[i, j, k] = scm[i, j]
        end
    end
    UpperTriangular(A)
end

"""
    downdate!(C::AbstractMatrix, A::AbstractMatrix)
    downdate!(C::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix)

Subtracts, in place, `A'A` or `A'B` from `C`
"""
function downdate! end

downdate!{T <: LinAlg.BlasFloat}(C::DenseMatrix{T}, A::DenseMatrix{T}) =
    BLAS.syrk!('U', 'T', -one(T), A, one(T), C)

downdate!{T <: LinAlg.BlasFloat}(C::DenseMatrix{T}, A::DenseMatrix{T}, B::DenseMatrix{T}) =
    BLAS.gemm!('T', 'N', -one(T), A, B, one(T), C)

function downdate!{T}(C::Diagonal{T}, A::SparseMatrixCSC{T})
    m, n = size(A)
    dd = C.diag
    if length(dd) ≠ n
        throw(DimensionMismatch("size(C,2) ≠ size(A,2)"))
    end
    nz = nonzeros(A)
    for j in eachindex(dd)
        for k in nzrange(A, j)
            @inbounds dd[j] -= abs2(nz[k])
        end
    end
    C
end

function downdate!{T}(C::Diagonal{T},A::Diagonal{T})
    if size(C) ≠ size(A)
        throw(DimensionMismatch("size(C) ≠ size(A)"))
    end
    c, a = C.diag, A.diag
    for i in eachindex(c)
        c[i] -= abs2(a[i])
    end
    C
end

function downdate!{T}(C::Diagonal{T}, A::Diagonal{T}, B::Diagonal{T})
    if !(size(C) == size(A) == size(B))
        throw(DimensionMismatch("need size(C) == size(A) == size(B)"))
    end
    c, a, b = C.diag, A.diag, B.diag
    for i in eachindex(c)
        c[i] -= a[i] * b[i]
    end
    C
end

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
    for jj in 1:s, j in 1:n, k in nzrange(A,j)
        C[j,jj] -= nz[k] * B[rv[k], jj]
    end
    C
end

function downdate!{T}(C::DenseMatrix{T}, A::SparseMatrixCSC{T}, B::SparseMatrixCSC{T})
    AtB = A'B
    if size(C) ≠ size(AtB)
        throw(DimensionMismatch("size(C) ≠ size(A'B)"))
    end
    atbrv = rowvals(AtB)
    atbnz = nonzeros(AtB)
    for j in 1:size(AtB, 2)
        for k in nzrange(AtB, j)
            C[atbrv[k], j] -= atbnz[k]
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
    if n ≠ LinAlg.checksquare(C)
        throw(DimensionMismatch("C is not square or size(C,2) ≠ size(A,2)"))
    end
    # FIXME: avoid allocation by caching a transposed matrix and just fill in the new values
    # alternatively, work with the lower Cholesky factor L instead of R
    At = A'
    rv = rowvals(A)
    nz = nonzeros(A)
    rvt = rowvals(At)
    nzt = nonzeros(At)
    cp = A.colptr
    @inbounds for j in 1:n
        for jp in nzrange(A, j)
            nzB = nz[jp]
            k = rv[jp]
            for kp in nzrange(At, k)
                C[rvt[kp], j] -= nzt[kp] * nzB
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

function inflate!(A::HBlkDiag)
    Aa = A.arr
    r, s, k = size(Aa)
    for j in 1 : k, i in 1 : min(r, s)
        Aa[i, i, j] += 1
    end
    A
end
function inflate!{T<:AbstractFloat}(A::StridedMatrix{T})
    n = LinAlg.checksquare(A)
    for i in 1 : n
        @inbounds A[i, i] += one(T)
    end
    A
end
function inflate!{T <: AbstractFloat}(D::Diagonal{T})
    d = D.diag
    for i in eachindex(d)
        d[i] += one(T)
    end
    D
end
