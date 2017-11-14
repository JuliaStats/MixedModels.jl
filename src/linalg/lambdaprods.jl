"""
    A_mul_Λ!(A::AbstractArray, B::AbstractTerm)

In-place product of `A` with a repeated block diagonal expansion of `B.Λ``

An [`AbstractTerm`]{@ref} of size n×k includes a description of a lower triangular
k×k matrix determined by the θ parameter vector.  These matrices are, at most, repeated
block diagonal (i.e. block diagonal with identical diagonal blocks).  The Λ block associated
with a [`MatrixTerm`]{@ref} is the identity.

See also [`scaleinflate!`]{@ref} which performs two such multiplications plus inflation of
the diagonal plus a copy! operation in one step.
"""
function A_mul_Λ! end
A_mul_Λ!(A, B::MatrixTerm) = A
A_mul_Λ!(A, B::ScalarFactorReTerm) = scale!(A, B.Λ)
function A_mul_Λ!(A::BlockedSparse{T}, B::VectorFactorReTerm{T}) where T
    λ = B.Λ
    for blk in A.colblocks
        A_mul_B!(blk, λ)
    end
    A
end
function A_mul_Λ!(A::Matrix{T}, B::VectorFactorReTerm{T,V,R,S}) where {T,V,R,S}
    λ = B.Λ
    m, n = size(A)
    q, r = divrem(n, S)
    iszero(r) || throw(DimensionMismatch("size(A, 2) = $n is not a multiple of S = $S"))
    inds = 1:S
    for blk in 1:q
        ## another place where ~ 1GB is allocated in d3 fit
        A_mul_B!(view(A, :, inds), λ)
        inds += S
    end
    A
end

"""
    Λc_mul_B!(A::AbstractTerm, B::AbstractArray)

In-place product of a repeated block diagonal expansion of `A.Λ'` with `B`

See also [`scaleinflate!`]{@ref} which performs two such multiplications plus inflation of
the diagonal plus a copy! operation in one step.
"""
function Λc_mul_B! end
Λc_mul_B!(A::MatrixTerm, B) = B
Λc_mul_B!(A::ScalarFactorReTerm, B) = scale!(A.Λ, B)
function Λc_mul_B!(A::VectorFactorReTerm{T,V,R,S}, B::Matrix{T}) where {T,V,R,S}
    m, n = size(B)
    Ac_mul_B!(A.Λ, reshape(B, (S, div(m, S) * n)))
    B
end

function Λc_mul_B!(A::VectorFactorReTerm{T}, B::BlockedSparse{T}) where T
    Ac_mul_B!(A.Λ, B.nzsasmat)
    B
end

"""
    Λ_mul_B!(C::Matrix, A::AbstractFactorReTerm, B::Matrix)

Mutating product of the repeated block-diagonal expansion of `A` and `B` into `C`
This multiplication is used to convert "spherical" random effects to the original scale.
"""
function Λ_mul_B!(C::Matrix, A::AbstractFactorReTerm, B::Matrix) end

function Λ_mul_B!(C::Matrix{T}, A::ScalarFactorReTerm{T}, B::Matrix{T}) where T
    @argcheck(size(C) == size(B) == (1, size(A, 2)), DimensionMismatch)
    scale!(C, A.Λ, B)
end

function Λ_mul_B!(C::Matrix{T}, A::VectorFactorReTerm{T,V,R,S}, B::Matrix{T}) where {T,V,R,S}
    @argcheck(size(C) == size(B) == (S, div(size(A, 2), S)), DimensionMismatch)
    A_mul_B!(A.Λ, copy!(C, B))
end
