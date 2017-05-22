# Products with respect to Λ blocks for various terms.

# The Λ block for a MatrixTerm is the identity
Ac_mul_B!{T}(A::MatrixTerm{T}, B::AbstractArray{T}) = B
A_mul_B!{T}(A::AbstractArray{T}, B::MatrixTerm{T}) = A

function A_mul_B!{T<:AbstractFloat}(A::Diagonal{LowerTriangular{T, Matrix{T}}},
    B::FactorReTerm{T})
    λ = LowerTriangular(B.Λ)
    for a in A.diag
        A_mul_B!(a.data, λ)
    end
    A
end

function A_mul_B!{T<:AbstractFloat,S}(A::SparseMatrixCSC{T,S}, B::FactorReTerm{T})
    λ = LowerTriangular(B.Λ)
    k = size(λ, 2)
    n = size(A, 2)
    rv = rowvals(A)
    nz = nonzeros(A)
    offset = 0
    while offset < n
        i1 = nzrange(A, offset + 1)
        rv1 = view(rv, i1)
        for j in 2:k
            all(rv1 .== view(rv, nzrange(A, offset + j))) || error("A is not compatible with B")
        end
        a = reshape(view(nz, i1.start:nzrange(A, offset + k).stop), (length(i1), k))
        A_mul_B!(a, a, λ)
        offset += k
    end
    A
end

function A_mul_B!{T<:AbstractFloat}(A::FactorReTerm{T}, B::StridedVector{T})
    λ = LowerTriangular(A.Λ)
    k = size(λ, 1)
    A_mul_B!(λ, reshape(B, (k, div(length(B), k))))
    B
end

function A_mul_B!{T<:AbstractFloat}(A::Matrix{T}, B::FactorReTerm{T})
    λ = LowerTriangular(B.Λ)
    k = size(λ, 1)
    m, n = size(A)
    q, r = divrem(n, k)
    if r ≠ 0
        throw(DimensionMismatch("size(A, 2) = $n is not a multiple of size(B.λ, 1) = $k"))
    end
    offset = 0
    onetok = 1:k
    for blk in 1:q
        A_mul_B!(view(A, :, onetok + offset), λ)
        offset += k
    end
    A
end

function A_mul_B!{T}(C::StridedVecOrMat{T}, A::FactorReTerm{T}, B::StridedVecOrMat{T})
    @argcheck(size(C) == size(B), DimensionMismatch)
    m = size(C, 1)
    λ = LowerTriangular(A.Λ)
    k = size(λ, 1)
    A_mul_B!(λ, reshape(copy!(C, B), (k, size(C, 2) * div(m, k))))
    C
end

function Ac_mul_B!{T}(A::FactorReTerm{T}, B::Diagonal{LowerTriangular{T,Matrix{T}}})
    λ = LowerTriangular(A.Λ)
    for b in B.diag
        Ac_mul_B!(λ, b.data)
    end
    B
end

function Ac_mul_B!{T}(A::FactorReTerm{T}, B::StridedVecOrMat{T})
    λ = LowerTriangular(A.Λ)
    k = size(λ, 1)
    m, n = size(B, 1), size(B, 2)
    Ac_mul_B!(λ, reshape(B, (k, div(m, k) * n)))
    B
end

function Ac_mul_B!{T<:AbstractFloat,S}(A::FactorReTerm{T}, B::SparseMatrixCSC{T,S})
    λ = LowerTriangular(A.Λ)
    k = size(λ, 2)
    nz = nonzeros(B)
    for j in 1:B.n
        bnz = view(nz, nzrange(B, j))
        mbj = reshape(bnz, (k, div(length(bnz), k)))
        Ac_mul_B!(mbj, λ, mbj)
    end
    B
end
