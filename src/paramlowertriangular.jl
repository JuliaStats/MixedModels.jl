nlower(n::Integer) = (n * (n + 1)) >>> 1
nlower{T}(A::LowerTriangular{T, Matrix{T}}) = nlower(Compat.LinAlg.checksquare(A))

"""
    getθ!{T}(v::AbstractVector{T}, A::LowerTriangular{T, Matrix{T}})

Overwrite `v` with the elements of the lower triangle of `A` (column-major ordering)
"""
function getθ!{T}(v::AbstractVector{T}, A::LowerTriangular{T,Matrix{T}})
    n, Ad = Compat.LinAlg.checksquare(A), A.data
    if length(v) ≠ nlower(n)
        throw(DimensionMismatch("length(v) = $(length(v)) should be $(nlower(n))"))
    end
    k = 0
    for j = 1 : n, i in j : n
        v[k += 1] = Ad[i, j]
    end
    v
end

"""
    getθ(A::LowerTriangular{T, Matrix{T}})

Return a vector of the elements of the lower triangle of `A` (column-major ordering)
"""
getθ{T}(A::LowerTriangular{T, Matrix{T}}) = getθ!(Array(T, nlower(A)), A)

"""
    lowerbd{T}(A::LowerTriangular{T,Matrix{T}})

Return the vector of lower bounds on the parameters, `θ`.

These are the elements in the lower triangle in column-major ordering.
Diagonals have a lower bound of `0`.  Off-diagonals have a lower-bound of `-Inf`.
"""
function lowerbd{T}(A::LowerTriangular{T,Matrix{T}})
    n = Compat.LinAlg.checksquare(A)
    res = fill(convert(T, -Inf), nlower(n))
    k = -n
    for j in n+1:-1:2
        res[k += j] = zero(T)
    end
    res
end

"""
    LT(A)

Create a lower triangular matrix compatible with the blocks of `A`
and initialized to the identity.
"""
LT{T}(A::ScalarReMat{T}) = LowerTriangular(ones(T, (1, 1)))
LT{T}(A::VectorReMat{T}) = LowerTriangular(full(eye(T, size(A.z, 1))))

function setθ!{T}(A::LowerTriangular{T, Matrix{T}}, v::AbstractVector{T})
    Ad = A.data
    n = Compat.LinAlg.checksquare(Ad)
    if length(v) ≠ nlower(n)
        throw(DimensionMismatch("length(v) = $(length(v)) should be $(nlower(n))"))
    end
    offset = 0
    for j in 1 : n, i in j : n
        Ad[i, j] = v[offset += 1]
    end
    A
end

"""
    tscale!(A::LowerTriangular, B::HBlkDiag)
    tscale!(A::LowerTriangular, B::Diagonal)
    tscale!(A::LowerTriangular, B::DenseVecOrMat)
    tscale!(A::LowerTriangular, B::SparseMatrixCSC)
    tscale!(A::HBlkDiag, B::LowerTriangular)
    tscale!(A::Diagonal, B::LowerTriangular)
    tscale!(A::DenseVecOrMat, B::LowerTriangular)
    tscale!(A::SparseMatrixCSC, B::LowerTriangular)

Scale a matrix using the implicit expansion of the lower triangular matrix
to a diagonal or homogeneous block diagonal matrix.

Used in evaluating `Λ'Z'ZΛ` from `Z'Z` without explicitly evaluating the matrix `Λ`.
"""
function tscale!(A::LowerTriangular,B::HBlkDiag)
    Ba = B.arr
    r, s, k = size(Ba)
    n = Compat.LinAlg.checksquare(A)
    if n ≠ r
        throw(DimensionMismatch("size(A,2) ≠ blocksize of B"))
    end
    Ac_mul_B!(A, reshape(Ba,(r,s * k)))
    B
end

function tscale!{T}(A::LowerTriangular{T}, B::Diagonal{T})
    if size(A, 1) ≠ 1
        throw(DimensionMismatch("A must be a 1×1 LowerTriangular"))
    end
    scale!(A.data[1], B.diag)
    B
end

function tscale!{T}(A::LowerTriangular{T}, B::DenseVecOrMat{T})
    if (l = size(A, 1)) == 1
        return scale!(A.data[1], B)
    end
    m, n = size(B, 1), size(B, 2)  # this sets n = 1 when B is a vector
    q, r = divrem(m, l)
    if r ≠ 0
        throw(DimensionMismatch("size(B,1) is not a multiple of size(A,1)"))
    end
    Ac_mul_B!(A, reshape(B, (l, q * n)))
    B
end

function tscale!{T}(A::LowerTriangular{T}, B::SparseMatrixCSC{T})
    if (l = size(A, 1)) == 1
        scale!(A.data[1], B.nzval)
    else
        m, n = size(B)
        q, r = divrem(m, l)
        if r ≠ 0
            throw(DimensionMismatch("size(B, 1) is not a multiple of size(A, 1)"))
        end
        q, r = divrem(nnz(B), l)
        if r ≠ 0
            throw(DimensionMismatch("nnz(B) is not a multiple of size(A, 1)"))
        end
        Ac_mul_B!(A, reshape(B.nzval, (l, q)))
    end
    B
end

function tscale!{T}(A::SparseMatrixCSC{T}, B::LowerTriangular{T})
    if (l = size(B, 1)) == 1
        scale!(A.nzval, B.data[1])
    else
        m, n = size(A)
        q, r = divrem(nnz(A), l)
        if r ≠ 0
            throw(DimensionMismatch("nnz(A) is not a multiple of size(B, 1)"))
        end
        q, r = divrem(n, l)
        if r ≠ 0
            throw(DimensionMismatch("size(A, 2) is not a multiple of size(B, 1)"))
        end
        Ar = rowvals(A)
        Acp = A.colptr
        Anz = nonzeros(A)
        offset = 0
        for k in 1 : q
            nzr1 = nzrange(A, offset + 1)
            Ar1 = Compat.view(Ar, nzr1)
            for j in 2 : l
                if Compat.view(Ar, nzrange(A, offset + j)) ≠ Ar1
                    throw(ArgumentError("A does not have block structure for tscale!"))
                end
            end
            lnzr = length(Ar1)
            Aa = reshape(Compat.view(Anz, nzr1[1] + (0 : (lnzr * l - 1))), (lnzr, l))
            A_mul_B!(Aa, Aa, B)
            offset += l
        end
    end
    A
end

function tscale!{T}(A::Diagonal{T}, B::LowerTriangular{T})
    if (l = Compat.LinAlg.checksquare(B)) ≠ 1
        throw(DimensionMismatch(
        "in tscale!(A::Diagonal,B::LowerTriangular) B must be 1×1"))
    end
    scale!(B.data[1], A.diag)
    A
end

function tscale!{T}(A::HBlkDiag{T}, B::LowerTriangular{T})
    aa = A.arr
    r, s, l = size(aa)
    scr = Array(T, r, s)
    for k in 1 : l
        for j in 1 : s, i in 1 : r
            scr[i, j] = aa[i, j, k]
        end
        A_mul_B!(scr, B)
        for j in 1 : s, i in 1 : r
            aa[i, j, k] = scr[i, j]
        end
    end
    A
end

function tscale!{T}(A::StridedMatrix{T}, B::LowerTriangular{T})
    if (l = size(B,1)) == 1
        return scale!(A, B.data[1])
    end
    m, n = size(A)
    q, r = divrem(n, l)
    if r ≠ 0
        throw(DimensionMismatch("size(A,2) = $n must be a multiple of size(B,1) = $l"))
    end
    for k in 0:(q - 1)
        A_mul_B!(Compat.view(A, : , k * l + (1:l)), B)
    end
    A
end
