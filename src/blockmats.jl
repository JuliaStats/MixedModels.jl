"""
`HBlkDiag` - a homogeneous block diagonal matrix, i.e. all the diagonal blocks are the same size

A matrix consisting of k diagonal blocks of size `r×s` is stored as an `r×s×k` array.
"""
immutable HBlkDiag{T} <: AbstractMatrix{T}
    arr::Array{T,3}
end

Base.eltype{T}(A::HBlkDiag{T}) = T

Base.size(A::HBlkDiag) = ((r,s,k) = size(A.arr); (r*k,s*k))

function Base.size(A::HBlkDiag,i::Integer)
    i < 1 && throw(BoundsError())
    i > 2 && return 1
    r,s,k = size(A.arr)
    (i == 1 ? r : s)*k
end

Base.copy!{T}(d::HBlkDiag{T},s::HBlkDiag{T}) = (copy!(d.arr,s.arr); d)

Base.copy{T}(s::HBlkDiag{T}) = HBlkDiag(copy(s.arr))

function Base.LinAlg.A_ldiv_B!(R::DenseVecOrMat,A::HBlkDiag,B::DenseVecOrMat)
    Aa = A.arr
    r,s,k = size(Aa)
    r == s || throw(ArgumentError("A must be square"))
    (m = size(B,1)) == size(R,1) || throw(DimensionMismatch())
    (n = size(B,2)) == size(R,2) || throw(DimensionMismatch())
    r*k == m || throw(DimensionMismatch())
    if r == 1
        for j in 1:n, i in 1:m
            R[i,j] = B[i,j]/Aa[i]
        end
    else
        db = similar(A.arr,(r,r))       # will hold the diagonal blocks
        for b in 1:k
            copy!(db,sub(A.arr,:,:,b))
            rows = (1:r)+(b-1)*r
            rr = copy!(sub(R,rows,:),sub(B,rows,:))
            Base.LinAlg.A_ldiv_B!(ishermitian(db) ? cholfact!(db) : lufact!(db), rr)
        end
    end
    R
end

function Base.getindex{T}(A::HBlkDiag{T},i::Integer,j::Integer)
    Aa = A.arr
    r,s,k = size(Aa)
    bi,ri = divrem(i-1,r)
    bj,rj = divrem(j-1,s)
    bi == bj || return zero(T)
    Aa[ri+1,rj+1,bi+1]
end

function Base.cholfact!(A::HBlkDiag,uplo::Symbol=:U)
    Aa = A.arr
    r,s,k = size(Aa)
    r == s || throw(ArgumentError("A must be square"))
    if r == 1
        for j in 1:k
            Aa[1,1,j] = sqrt(Aa[1,1,j])
        end
    else
        for j in 1:k
            cholfact!(sub(Aa,:,:,j),uplo)
        end
    end
    A
end

function Base.full(A::HBlkDiag)
    aa = A.arr
    res = zeros(eltype(aa),size(A))
    p,q,l = size(aa)
    for b in 1:l
        bm1 = b - 1
        for j in 1:q
            for i in 1:p
                res[bm1*p+i,bm1*q+j] = aa[i,j,b]
            end
        end
    end
    res
end

"""
Equivalent to `A = A + I` without making a copy of A
"""
function inflate!(A::HBlkDiag)
    Aa = A.arr
    r,s,k = size(Aa)
    for j in 1:k, i in 1:min(r,s)
        Aa[i,i,j] += 1
    end
    A
end
