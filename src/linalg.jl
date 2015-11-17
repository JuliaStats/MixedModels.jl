function Base.LinAlg.Ac_ldiv_B!{T}(D::Diagonal{T},B::DenseMatrix{T})
    m,n = size(B)
    dd = D.diag
    if length(dd) ≠ m
        throw(DimensionMismatch("size(D,2) ≠ size(B,1)"))
    end
    for j in 1:n, i in 1:m
        B[i,j] /= dd[i]
    end
    B
end

function Base.LinAlg.Ac_ldiv_B!{T}(D::Diagonal{T},B::Diagonal{T})
    dd = D.diag
    bd = B.diag
    if length(dd) ≠ length(bd)
        throw(DimensionMismatch("size(D,2) ≠ size(B,1)"))
    end
    for j in eachindex(bd)
        bd[j] /= dd[j]
    end
    B
end

function Base.LinAlg.Ac_ldiv_B!{T}(A::UpperTriangular{T,HBlkDiag{T}},B::DenseMatrix{T})
    m,n = size(B)
    aa = A.data.arr
    r,s,k = size(aa)
    if m ≠ Base.LinAlg.chksquare(A)
        throw(DimensionMismatch("size(A,2) ≠ size(B,1)"))
    end
    scr = Array(T,(r,n))
    for i in 1:k
        bb = sub(B,(i-1)*r+(1:r),:)
        copy!(bb,Base.LinAlg.Ac_ldiv_B!(UpperTriangular(sub(aa,:,:,i)),copy!(scr,bb)))
    end
    B
end

function Base.LinAlg.Ac_ldiv_B!{T}(D::Diagonal{T},B::SparseMatrixCSC{T})
    m,n = size(B)
    dd = D.diag
    if length(dd) ≠ m
        throw(DimensionMismatch("size(D,2) ≠ size(B,1)"))
    end
    nzv = nonzeros(B)
    rv = rowvals(B)
    for j in 1:n, k in nzrange(B,j)
        nzv[k] /= dd[rv[k]]
    end
    B
end

## method not called in tests
Base.LinAlg.A_ldiv_B!{T<:AbstractFloat}(D::Diagonal{T},B::DenseMatrix{T}) =
    Base.LinAlg.Ac_ldiv_B!(D,B)

if false
## method not called in tests
function Base.LinAlg.A_rdiv_Bc!{T<:AbstractFloat}(
    A::SparseMatrixCSC{T},
    B::Diagonal{T}
    )
    m,n = size(A)
    dd = B.diag
    n == length(dd) || throw(DimensionMismatch())
    nz = nonzeros(A)
    for j in eachindex(dd)
        @inbounds scale!(sub(nz,nzrange(A,j)),inv(dd[j]))
    end
    A
end

## method not called in tests
function Base.LinAlg.A_rdiv_Bc!{T<:AbstractFloat}(A::Matrix{T},B::Diagonal{T})
    m,n = size(A)
    dd = B.diag
    n == length(dd) || throw(DimensionMismatch(""))
    for j in eachindex(dd)
        @inbounds scale!(sub(A,:,j),inv(dd[j]))
    end
    A
end

## method not called in tests
function Base.LinAlg.A_rdiv_B!{T}(A::StridedVecOrMat{T},D::Diagonal{T})
    m, n = size(A, 1), size(A, 2)
    if n ≠ length(D.diag)
        throw(DimensionMismatch("diagonal matrix is $(length(D.diag)) by $(length(D.diag)) but left hand side has $n columns"))
    end
    if m == 0 || n == 0
        return A
    end
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

end
