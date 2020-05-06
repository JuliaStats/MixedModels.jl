"""
    cholUnblocked!(A, Val{:L})

Overwrite the lower triangle of `A` with its lower Cholesky factor.

The name is borrowed from [https://github.com/andreasnoack/LinearAlgebra.jl]
because these are part of the inner calculations in a blocked Cholesky factorization.
"""
function cholUnblocked! end

function cholUnblocked!(D::Diagonal{T}, ::Type{Val{:L}}) where {T<:AbstractFloat}
    map!(sqrt, D.diag, D.diag)
    D
end

function cholUnblocked!(A::StridedMatrix{T}, ::Type{Val{:L}}) where {T<:BlasFloat}
    n = LinearAlgebra.checksquare(A)
    if n == 1
        A[1] < zero(T) && throw(PosDefException(1))
        A[1] = sqrt(A[1])
    elseif n == 2
        A[1] < zero(T) && throw(PosDefException(1))
        A[1] = sqrt(A[1])
        A[2] /= A[1]
        (A[4] -= abs2(A[2])) < zero(T) && throw(PosDefException(2))
        A[4] = sqrt(A[4])
    else
        _, info = LAPACK.potrf!('L', A)
        iszero(info) || throw(PosDefException(info))
    end
    A
end

function cholUnblocked!(D::UniformBlockDiagonal, ::Type{Val{:L}})
    Ddat = D.data
    for k in axes(Ddat, 3)
        cholUnblocked!(view(Ddat, :, :, k), Val{:L})
    end
    D
end

# The following are based on https://arxiv.org/abs/1602.07527,
# "Differentiation of the Cholesky decomposition" by Iain Murray

"""
    level2partition(A, j)

Return the partition r, d, B, c of views forming the lower-left block anchored at `A[j,j]`
"""
function level2partition(A::AbstractMatrix, j)
    N = LinearAlgebra.checksquare(A)
    checkbounds(Bool, axes(A, 1), j) || throw(ArgumentError("j=$j must be in 1:$N"))
    view(A, j, 1:(j-1)), A[j, j], view(A, (j+1):N, 1:(j-1)), view(A, (j+1):N, j)
end

"""
    chol_unblocked_fwd!(Σ̇::AbstractMatrix, L::AbstractMatrix)

Overwrite the sensitivities, Σ̇, by the sensitivities, L̇, of the lower Cholesky factor, L

This is the scalar, unblocked version of the forward mode differentiation of the Cholesky
factor given at the bottom of p. 6 in the reference.  Only the lower triangle of Σ̇ is
referenced and overwritten.

Note how close this code is to the pseudo-code in the reference - much closer than the
Python code in the appendix.
"""
function chol_unblocked_fwd!(Ȧ::AbstractMatrix{T}, L::AbstractMatrix{T}) where {T<:Real}
    if LinearAlgebra.checksquare(Ȧ) ≠ LinearAlgebra.checksquare(L)
        throw(DimensionMismatch("Ȧ and L must be square and of the same size"))
    end
    for j in axes(Ȧ, 2)
        r, d, B, c = level2partition(L, j)
        ṙ, ḋ, Ḃ, ċ = level2partition(Ȧ, j)
        Ȧ[j,j] = ḋ = (ḋ/2 - dot(r, ṙ)) / d
        ċ .= (ċ - Ḃ*r - B*ṙ - c*ḋ) / d
    end
    Ȧ
end

function chol_unblocked_fwd(Σ̇::AbstractMatrix{T}, L::AbstractMatrix{T}) where {T<:Real}
    chol_unblocked_fwd!(copy(Σ̇), L)
end

function chol_unblocked!(A::AbstractMatrix{<:Real})
    LinearAlgebra.checksquare(A)
    for j in axes(A, 2)
        r, d, B, c = level2partition(A, j)
        A[j, j] = d = sqrt(d - sum(abs2, r))
        invd = inv(d)
        mul!(c, B, r, -invd, invd)
        c 
    end
    A
end

function chol_unblocked_and_fwd!(Ȧ::Vector{T}, A::T) where {T<:AbstractMatrix{<:Real}}
    if !all(isequal(LinearAlgebra.checksquare(A)), LinearAlgebra.checksquare.(Ȧ))
        throw(DimensionMismatch("A and elements of Ȧ must be square and the same size"))
    end
    for j in axes(A, 2)
        r, d, B, c = level2partition(A, j)
        A[j, j] = d = sqrt(d - sum(abs2, r))
        invd = inv(d)
        mul!(c, B, r, -invd, invd)
        for Ȧk in Ȧ
            ṙ, ḋ, Ḃ, ċ = level2partition(Ȧk, j)
            Ȧk[j, j] = ḋ = (ḋ/2 - dot(r, ṙ)) / d
            ċ .= (ċ .- Ḃ*r .- B*ṙ .- c*ḋ) ./ d
        end
    end
    Ȧ, A
end
