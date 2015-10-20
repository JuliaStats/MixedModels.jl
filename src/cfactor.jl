"""
Slightly modified version of `chol!` from `julia/base/linalg/cholesky.jl`

Uses `inject!` (as opposed to `copy!`), `downdate!` (as opposed to `syrk!`
    or `gemm!`) and recursive calls to `cfactor!`,
"""
function cfactor!(A::AbstractMatrix)
    n = Base.LinAlg.chksquare(A)
    for k = 1:n
        Akk = A[k,k]
        for i in 1:(k - 1)
            downdate!(Akk,A[i,k])    # A[k,k] -= A[i,k]'A[i,k]
        end
        Akk = cfactor!(Akk)          # right Cholesky factor of A[k,k]
        for j in (k + 1):n
            for i in 1:(k - 1)
                downdate!(A[k,j],A[i,k],A[i,j]) # A[k,j] -= A[i,k]'A[i,j]
            end
            Base.LinAlg.Ac_ldiv_B!(Akk,A[k,j])
        end
    end
    UpperTriangular(A)
end

function cfactor!(D::Diagonal)
    map!(sqrt,D.diag)
    UpperTriangular(D)
end

cfactor!(R::Matrix{Float64}) = UpperTriangular(Base.LinAlg.LAPACK.potrf!('U',R)[1])

function cfactor!(R::HBlkDiag)
    Ra = R.arr
    r,s,k = size(Ra)
    for i in 1:k
        Base.LinAlg.chol!(sub(Ra,:,:,i),Val{:U})
    end
    UpperTriangular(R)
end

"""
Subtract, in place, A'A or A'B from C
"""
downdate!{T<:Base.LinAlg.BlasFloat}(C::DenseMatrix{T},A::DenseMatrix{T}) =
    BLAS.syrk!('U','T',-one(T),A,one(T),C)

downdate!{T<:Base.LinAlg.BlasFloat}(C::DenseMatrix{T},A::DenseMatrix{T},B::DenseMatrix{T}) =
    BLAS.gemm!('T','N',-one(T),A,B,one(T),C)

function downdate!{T}(C::Diagonal{T},A::SparseMatrixCSC{T})
    m,n = size(A)
    dd = C.diag
    length(dd) == n || throw(DimensionMismatch(""))
    nz = nonzeros(A)
    for j in eachindex(dd)
        for k in nzrange(A,j)
            @inbounds dd[j] -= abs2(nz[k])
        end
    end
    C
end

function downdate!{T}(C::Diagonal{T},A::Diagonal{T})
    size(C) == size(A) || throw(DimensionMismatch())
    c = C.diag
    a = A.diag
    for i in eachindex(c)
        c[i] -= abs2(a[i])
    end
    C
end

function downdate!{T}(C::Diagonal{T},A::Diagonal{T},B::Diagonal{T})
    size(C) == size(A) == size(B) || throw(DimensionMismatch())
    c,a,b = C.diag,A.diag,B.diag
    for i in eachindex(c)
        c[i] -= a[i]*b[i]
    end
    C
end

function downdate!{T}(C::DenseMatrix{T},A::Diagonal{T},B::DenseMatrix{T})
    a = A.diag
    ((m,n) = size(B)) == size(C) || throw(DimensionMismatch())
    length(a) == m || throw(DimensionMismatch())
    for j in 1:n, i in 1:m
        C[i,j] -= a[i] * B[i,j]
    end
    C
end

function downdate!{T}(C::DenseMatrix{T},A::SparseMatrixCSC{T},B::DenseMatrix{T})
    m,n = size(A)
    r,s = size(C)
    r == n && s == size(B,2) && m == size(B,1) || throw(DimensionMismatch(""))
    nz = nonzeros(A)
    rv = rowvals(A)
    for jj in 1:s, j in 1:n, k in nzrange(A,j)
        C[j,jj] -= nz[k]*B[rv[k],jj]
    end
    C
end

function downdate!{T}(C::DenseMatrix{T},A::SparseMatrixCSC{T},B::SparseMatrixCSC{T})
    AtB = A'B
    size(C) == size(AtB) || throw(DimensionMismatch())
    atbrv = rowvals(AtB)
    atbnz = nonzeros(AtB)
    for j in 1:size(AtB,2)
        for k in nzrange(AtB,j)
            C[atbrv[k],j] -= atbnz[k]
        end
    end
    C
end

function downdate!{T}(C::DenseMatrix{T},A::SparseMatrixCSC{T})
    m,n = size(A)
    n == Base.LinAlg.chksquare(C) || throw(DimensionMismatch(""))
    tt = A'A
    nzv = nonzeros(tt)
    rv = rowvals(tt)
    for j in 1:n
        for k in nzrange(tt,j)
            if (i = rv[k]) ≤ j
                C[i,j] -= nzv[k]
            end
        end
    end
    C
end

function downdate!{T}(C::DenseMatrix{T},A::DenseMatrix{T},B::SparseMatrixCSC{T})
    ma,na = size(A)
    mb,nb = size(B)
    ma == size(C,1) && mb == size(C,2) && na == nb || throw(DimensionMismatch(""))
    rvb = rowvals(B); nzb = nonzeros(B)
    for j in 1:nb
        ptA = pointer(A,1+(j-1)*ma)
        ib = nzrange(B,j)
        rvbj = sub(rvb,ib)
        nzbj = sub(nzb,ib)
        for k in eachindex(ib)
            BLAS.axpy!(ma,-nzbj[k],ptA,1,pointer(C,1+(rvbj[k]-1)*ma),1)
        end
        ptA += ma
    end
    C
end
