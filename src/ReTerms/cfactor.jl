"""
Slightly modified version of `chol!` from `julia/base/linalg/cholesky.jl`

Uses `inject!` (as opposed to `copy!`), `downdate!` (as opposed to `syrk!`
    or `gemm!`) and recursive calls to `cfactor!`,
"""
function cfactor!(A::AbstractMatrix)
    n = Base.LinAlg.chksquare(A)
#    @inbounds begin
        for k = 1:n
            Akk = A[k,k]
            for i in 1:(k - 1)
                downdate!(Akk,A[i,k])  # A[k,k] -= A[i,k]'A[i,k]
            end
            Akk = cfactor!(Akk)   # right Cholesky factor of A[k,k]
            for j in (k + 1):n
                for i in 1:(k - 1)
                    downdate!(A[k,j],A[i,k],A[i,j]) # A[k,j] -= A[i,k]*A[i,j]
                end
                Base.LinAlg.Ac_ldiv_B!(Akk,A[k,j])
            end
        end
#    end
    UpperTriangular(A)
end

function cfactor!(D::Diagonal)
    map!(sqrt,D.diag)
    UpperTriangular(D)
end

cfactor!(R::Matrix{Float64}) = Base.LinAlg.chol!(R,Val{:U})

function cfactor!(R::HBlkDiag)
    Ra = R.arr
    r,s,k = size(Ra)
    for i in 1:k
        Base.LinAlg.chol!(sub(Ra,:,:,i),Val{:U})
    end
    UpperTriangular(R)
end

"Subtract, in place, A'A or A'B from C"
downdate!(C::DenseMatrix{Float64},A::DenseMatrix{Float64}) =
    BLAS.syrk!('U','T',-1.0,A,1.0,C)
downdate!(C::DenseMatrix{Float64},A::DenseMatrix{Float64},B::DenseMatrix{Float64}) =
    BLAS.gemm!('T','N',-1.0,A,B,1.0,C)
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
    ma,na = size(A)
    mb,nb = size(B)
    ma == size(C,1) && mb == size(C,2) && na == nb || throw(DimensionMismatch(""))
    rva = rowvals(A); nza = nonzeros(A); rvb = rowvals(B); nzb = nonzeros(B)
    for j in 1:nb
        ia = nzrange(A,j)
        ib = nzrange(B,j)
        rvaj = sub(rva,ia)
        rvbj = sub(rvb,ib)
        nzaj = sub(nza,ia)
        nzbj = sub(nzb,ib)
        for k in eachindex(ib)
            krv = rvbj[k]
            knz = nzbj[k]
            for i in eachindex(ia)
                @inbounds C[rvaj[i],krv] -= nzaj[i]*knz
            end
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
