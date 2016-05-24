"""
    cfactor!(A::AbstractMatrix)
A slightly modified version of `chol!` from `julia/base/linalg/cholesky.jl`

Uses `inject!` (as opposed to `copy!`), `downdate!` (as opposed to `syrk!` or `gemm!`)
and recursive calls to `cfactor!`.

Note: The `cfactor!` method for dense matrices calls `LAPACK.potrf!` directly to avoid
errors being thrown when `A` is computationally singular
"""
function cfactor!(A::AbstractMatrix)
    n = Compat.LinAlg.checksquare(A)
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
    densify(S::SparseMatrix, threshold=0.3)
Convert sparse `S` to `Diagonal` if `S` is diagonal or to `full(S)` if
the proportion of nonzeros exceeds `threshold`.
"""
function densify(S::SparseMatrixCSC, threshold=0.3)
    m,n = size(S)
    if m == n && isdiag(S)  # convert diagonal sparse to Diagonal
        return Diagonal(diag(S))
    end
    if nnz(S)/(*(size(S)...)) ≤ threshold # very sparse matrices left as is
        return S
    end
    if isbits(eltype(S))
        return full(S)
    end
    # densify a sparse matrix whose elements are arrays of bitstypes
    nzs = nonzeros(S)
    nz1 = nzs[1]
    T = typeof(nz1)
    if !isa(nz1, Array) || !isbits(eltype(nz1)) # branch not tested
        error("Nonzeros must be a bitstype or an array of same")
    end
    sz1 = size(nz1)
    if any(x->typeof(x) ≠ T || size(x) ≠ sz1, nzs) # branch not tested
        error("Inconsistent dimensions or types in array nonzeros")
    end
    M,N = size(S)
    m,n = size(nz1, 1), size(nz1, 2) # this construction allows for nz1 to be a vector
    res = Array(eltype(nz1), M * m, N * n)
    rv = rowvals(S)
    for j in 1:size(S,2)
        for k in nzrange(S, j)
            copy!(sub(res, (rv[k] - 1) * m + (1 : m), (j - 1) * n + (1 : n)), nzs[k])
        end
    end
    res
end
densify(A::AbstractMatrix, threshold = 0.3) = A

"""
    downdate!(C::AbstractMatrix, A::AbstractMatrix)
    downdate!(C::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix)
Subtract, in place, `A'A` or `A'B` from `C`
"""
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
    m, n = size(A)
    r, s = size(C)
    if r ≠ n || s ≠ size(B,2) || m ≠ size(B,1)
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

function downdate!{T}(C::DenseMatrix{T}, A::SparseMatrixCSC{T})
    m, n = size(A)
    if n ≠ Compat.LinAlg.checksquare(C)
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
Equivalent to `A += I`, without making a copy of `A`.  Even if `A += I` did not
make a copy, this function is needed for the special behavior on the `HBlkDiag` type.
"""
function inflate!(A::HBlkDiag)
    Aa = A.arr
    r, s, k = size(Aa)
    for j in 1 : k, i in 1 : min(r, s)
        Aa[i, i, j] += 1
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
function inflate!{T<:AbstractFloat}(A::StridedMatrix{T})
    n = Compat.LinAlg.checksquare(A)
    for i in 1 : n
        @inbounds A[i, i] += one(T)
    end
    A
end
