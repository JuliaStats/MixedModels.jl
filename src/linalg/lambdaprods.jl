"""
    Λ_mul_B!(A::AbstractTerm, B::AbstractArray)
    Λc_mul_B!(A::AbstractTerm, B::AbstractArray)
    A_mul_Λ!(A::AbstractArray, B::AbstractTerm)
    
In-place products w.r.t. blocks of Λ or Λ′

An [`AbstractTerm`]{@ref} of size n×k includes a description of a lower triangular
k×k matrix determined by the θ parameter vector.  These matrices are, at most, repeated
block diagonal (i.e. block diagonal with identical diagonal blocks).  The Λ block associated
with a [`MatrixTerm`]{@ref} is the identity.

See also [`scaleinflate!`]{@ref} which performs two such multiplications plus inflation of
the diagonal plus a copy! operation in one step.
"""
function Λc_mul_B! end
function A_mul_Λ! end

Λc_mul_B!(A::MatrixTerm, B) = B
A_mul_Λ!(A, B::MatrixTerm) = A

A_mul_Λ!(A, B::ScalarFactorReTerm) = scale!(A, B.Λ)
Λc_mul_B!(A::ScalarFactorReTerm, B) = scale!(A.Λ, B)

function A_mul_Λ!(A::BlockedSparse{T}, B::VectorFactorReTerm{T}) where T
    λ = B.Λ
    for blk in A.colblocks
        A_mul_B!(blk, λ)
    end
    A
end

function Λ_mul_B!(A::VectorFactorReTerm{T}, B::StridedVector{T}) where T
    λ = A.Λ
    k = size(λ, 1)
    A_mul_B!(λ, reshape(B, (k, div(length(B), k))))
    B
end

Λ_mul_B!(A::ScalarFactorReTerm{T}, B::StridedVecOrMat{T}) where T = scale!(B, A.Λ)

function A_mul_Λ!(A::Matrix{T}, B::VectorFactorReTerm{T}) where T<:AbstractFloat
    @argcheck (k = vsize(B)) > 1
    λ = B.Λ
    m, n = size(A)
    q, r = divrem(n, k)
    if r ≠ 0
        throw(DimensionMismatch("size(A, 2) = $n is not a multiple of size(B.λ, 1) = $k"))
    end
    inds = 1:k
    for blk in 1:q
        ## another place where ~ 1GB is allocated in d3 fit
        A_mul_B!(view(A, :, inds), λ)
        inds += k
    end
    A
end

Λ_mul_B!(C::AbstractArray{T}, A::ScalarFactorReTerm{T}, B::AbstractArray{T}) where T = scale!(C, A.Λ, B)

function Λ_mul_B!(C::StridedVecOrMat{T}, A::VectorFactorReTerm{T},
                  B::StridedVecOrMat{T}) where T
    @argcheck(size(C) == size(B), DimensionMismatch)
    k = vsize(A)
    q, r = divrem(size(C, 1), k)
    iszero(r) || throw(ArgumentError("size(C, 1) = $(size(C,1)) is not a multiple of $k = vsize(A)"))
    A_mul_B!(A.Λ, reshape(copy!(C, B), (k, size(C, 2) * q)))
    C
end

function Λc_mul_B!(A::VectorFactorReTerm{T}, B::StridedVecOrMat{T}) where T
    @argcheck (k = vsize(A)) > 1
    λ = A.Λ
    m, n = size(B, 1), size(B, 2)
    Ac_mul_B!(λ, reshape(B, (k, div(m, k) * n)))
    B
end

function Λc_mul_B!(A::VectorFactorReTerm{T}, B::BlockedSparse{T}) where T
    Ac_mul_B!(A.Λ, B.nzsasmat)
    B
end
