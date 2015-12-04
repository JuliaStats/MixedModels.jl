nlower(n::Integer) = (n*(n+1))>>1
nlower{T}(A::LowerTriangular{T,Matrix{T}}) = nlower(Base.LinAlg.chksquare(A))

"""
return the lower triangle as a vector (column-major ordering)
"""
function Base.getindex{T}(A::LowerTriangular{T,Matrix{T}},s::Symbol)
    if s ≠ :θ
        throw(KeyError(s))
    end
    n = Base.LinAlg.chksquare(A)
    res = Array(T,nlower(n))
    k = 0
    for j = 1:n, i in j:n
        @inbounds res[k += 1] = A[i,j]
    end
    res
end

"""
set the lower triangle of A to v using column-major ordering
"""
function Base.setindex!{T}(
    A::LowerTriangular{T,Matrix{T}},
    v::AbstractVector{T},
    s::Symbol
    )
    if s ≠ :θ
        throw(KeyError(s))
    end
    n = Base.LinAlg.chksquare(A)
    if length(v) ≠ nlower(n)
        throw(DimensionMismatch("length(v) ≠ nlower(A)"))
    end
    k = 0
    for j in 1:n, i in j:n
        A[i,j] = v[k += 1]
    end
    A
end

"""
lower bounds on the parameters (elements in the lower triangle)
"""
function lowerbd{T}(A::LowerTriangular{T,Matrix{T}})
    n = Base.LinAlg.chksquare(A)
    res = fill(convert(T,-Inf),nlower(n))
    k = -n
    for j in n+1:-1:2
        res[k += j] = zero(T)
    end
    res
end

chksz(A::ScalarReMat,λ::LowerTriangular) = size(λ,1) == 1
chksz(A::VectorReMat,λ::LowerTriangular) = size(λ,1) == size(A.z,1)

"""
`rowlengths(L)` -> a vector of the Euclidean lengths of the rows of `L`

used in `chol2cor`
"""
function rowlengths(L::LowerTriangular)
    Ld = L.data
    [norm(sub(Ld,i,1:i)) for i in 1:size(Ld,1)]
end

"""
scale B using the implicit expansion of A to a homogeneous block diagonal
"""
function tscale!(A::LowerTriangular,B::HBlkDiag)
    Ba = B.arr
    r,s,k = size(Ba)
    n = Base.LinAlg.chksquare(A)
    if n ≠ r
        throw(DimensionMismatch("size(A,2) ≠ blocksize of B"))
    end
    Ac_mul_B!(A,reshape(Ba,(r,s*k)))
    B
end

function tscale!{T}(A::LowerTriangular{T},B::Diagonal{T})
    if Base.LinAlg.chksquare(A) ≠ 1
        throw(DimensionMismatch("A must be a 1×1 LowerTriangular"))
    end
    scale!(A.data[1],B.diag)
    B
end

"""
`LT(A) -> LowerTriangular`

Create as a lower triangular matrix compatible with the blocks of `A`
and initialized to the identity.
"""
LT(A::ScalarReMat) = LowerTriangular(ones(eltype(A.z),(1,1)))

function LT(A::VectorReMat)
    Az = A.z
    LowerTriangular(full(eye(eltype(Az),size(Az,1))))
end

function tscale!{T}(A::LowerTriangular{T},B::DenseVecOrMat{T})
    if (l = Base.LinAlg.chksquare(A)) == 1
        return scale!(A.data[1],B)
    end
    m,n = size(B,1),size(B,2)  # this sets n = 1 when B is a vector
    q,r = divrem(m,l)
    if r ≠ 0
        throw(DimensionMismatch("size(B,1) is not a multiple of size(A,1)"))
    end
    Ac_mul_B!(A,reshape(B,(l,q*n)))
    B
end

function tscale!{T}(A::LowerTriangular{T},B::SparseMatrixCSC{T})
    if (l = Base.LinAlg.chksquare(A)) ≠ 1
        error("Code not yet written")
    end
    scale!(A.data[1],B.nzval)
    B
end

function tscale!{T}(A::SparseMatrixCSC{T},B::LowerTriangular)
    if (l = Base.LinAlg.chksquare(B)) != 1
        error("Code not yet written")
    end
    scale!(A.nzval,B.data[1])
    A
end

function tscale!{T}(A::Diagonal{T},B::LowerTriangular{T})
    if (l = Base.LinAlg.chksquare(B)) ≠ 1
        throw(DimensionMismatch(
        "in tscale!(A::Diagonal,B::LowerTriangular) B must be 1×1"))
    end
    scale!(B.data[1],A.diag)
    A
end

function tscale!{T}(A::HBlkDiag{T},B::LowerTriangular{T})
    aa = A.arr
    r,s,k = size(aa)
    for i in 1:k
        A_mul_B!(sub(aa,:,:,i),B)
    end
    A
end

function tscale!{T}(A::StridedMatrix{T},B::LowerTriangular{T})
    l = Base.LinAlg.chksquare(B)
    l == 1 && return scale!(A,B.data[1])
    m,n = size(A)
    q,r = divrem(n,l)
    if r ≠ 0
        throw(DimensionMismatch("size(A,2) = $n must be a multiple of size(B,1) = $l"))
    end
    for k in 0:(q-1)
        A_mul_B!(sub(A,:,k*l + (1:l)),B)
    end
    A
end
