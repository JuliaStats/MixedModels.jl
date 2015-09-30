function Base.LinAlg.Ac_ldiv_B!{T}(D::UpperTriangular{T,Diagonal{T}},B::DenseMatrix{T})
    m,n = size(B)
    dd = D.data.diag
    length(dd) == m || throw(DimensionMismatch(""))
    for j in 1:n, i in 1:m
        B[i,j] /= dd[i]
    end
    B
end

function Base.LinAlg.Ac_ldiv_B!{T}(A::UpperTriangular{T,HBlkDiag{T}},B::DenseMatrix{T})
    m,n = size(B)
    aa = A.data.arr
    r,s,k = size(aa)
    m == Base.LinAlg.chksquare(A) || throw(DimensionMismatch())
    scr = Array(T,(r,n))
    for i in 1:k
        bb = sub(B,(i-1)*r+(1:r),:)
        copy!(bb,Base.LinAlg.Ac_ldiv_B!(UpperTriangular(sub(aa,:,:,i)),copy!(scr,bb)))
    end
    B
end

function Base.LinAlg.Ac_ldiv_B!{T}(D::UpperTriangular{T,Diagonal{T}},B::SparseMatrixCSC{T})
    m,n = size(B)
    dd = D.data.diag
    length(dd) == m || throw(DimensionMismatch(""))
    nzv = nonzeros(B)
    rv = rowvals(B)
    for j in 1:n, k in nzrange(B,j)
        nzv[k] /= dd[rv[k]]
    end
    B
end

Base.LinAlg.A_ldiv_B!{T<:AbstractFloat}(D::Diagonal{T},B::DenseMatrix{T}) =
    Base.LinAlg.Ac_ldiv_B!(D,B)

function Base.LinAlg.A_rdiv_Bc!{T<:AbstractFloat}(A::SparseMatrixCSC{T},B::Diagonal{T})
    m,n = size(A)
    dd = B.diag
    n == length(dd) || throw(DimensionMismatch(""))
    nz = nonzeros(A)
    for j in eachindex(dd)
        @inbounds scale!(sub(nz,nzrange(A,j)),inv(dd[j]))
    end
    A
end

function Base.LinAlg.A_rdiv_Bc!{T<:AbstractFloat}(A::Matrix{T},B::Diagonal{T})
    m,n = size(A)
    dd = B.diag
    n == length(dd) || throw(DimensionMismatch(""))
    for j in eachindex(dd)
        @inbounds scale!(sub(A,:,j),inv(dd[j]))
    end
    A
end

function Base.LinAlg.A_rdiv_B!(A::StridedVecOrMat,D::Diagonal)
    m, n = size(A, 1), size(A, 2)
    if n != length(D.diag)
        throw(DimensionMismatch("diagonal matrix is $(length(D.diag)) by $(length(D.diag)) but left hand side has $n columns"))
    end
    (m == 0 || n == 0) && return A
    dd = D.diag
    for j = 1:n
        dj = dd[j]
        if dj == 0
            throw(SingularException(j))
        end
        for i = 1:m
            A[i,j] /= dj
        end
    end
    A
end
