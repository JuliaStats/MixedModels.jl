"""
    cholUnblocked!(A, Val{:L})

Overwrite the lower triangle of `A` with its lower Cholesky factor.

The name is borrowed from [https://github.com/andreasnoack/LinearAlgebra.jl]
because these are part of the inner calculations in a blocked Cholesky factorization.
"""
function cholUnblocked! end

function cholUnblocked!(D::Diagonal{T}, ::Type{Val{:L}}) where {T<:AbstractFloat}
    Ddiag = D.diag
    @inbounds for i in eachindex(Ddiag)
        (ddi = Ddiag[i]) ≤ zero(T) && throw(PosDefException(i))
        Ddiag[i] = sqrt(ddi)
    end

    return D
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
    return A
end

function cholUnblocked!(D::UniformBlockDiagonal, ::Type{Val{:L}})
    Ddat = D.data
    for k in axes(Ddat, 3)
        cholUnblocked!(view(Ddat, :, :, k), Val{:L})
    end
    return D
end
