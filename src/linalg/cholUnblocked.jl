"""
    cholUnblocked!(A, Val{:L})

Overwrite the lower triangle of `A` with its lower Cholesky factor.

The name is borrowed from [https://github.com/andreasnoack/LinearAlgebra.jl]
because these are part of the inner calculations in a blocked Cholesky factorization.
"""
function cholUnblocked! end

function cholUnblocked!(D::Diagonal{T}, ::Type{Val{:L}}) where T<:AbstractFloat
    map!(sqrt, D.diag, D.diag)
    D
end

function cholUnblocked!(A::Matrix{T}, ::Type{Val{:L}}) where T<:BlasFloat
    n = checksquare(A)
    if n == 1
        A[1] < zero(T) && throw(PosDefException(1))
        A[1] = sqrt(A[1])
    elseif n == 2
        A[1] = sqrt(A[1])
        A[2] /= A[1]
        A[4] = sqrt(A[4] - abs2(A[2]))
    else
        _, info = LAPACK.potrf!('L', A)
        info ≠ 0 && throw(PosDefException(info))
    end
    A
end

function cholUnblocked!(D::Diagonal{LowerTriangular{T, Matrix{T}}},
::Type{Val{:L}}) where T<:AbstractFloat
    for b in D.diag
        cholUnblocked!(b.data, Val{:L})
    end
    D
end
