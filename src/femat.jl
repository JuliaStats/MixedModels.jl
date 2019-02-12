"""
    FeMat{T,S}

Term with an explicit, constant matrix representation

# Fields
* `x`: matrix
* `wtx`: weighted matrix
* `piv`: pivot `Vector{Int}`` for pivoted Cholesky factorization of `wtx'wtx`
* `rank`: computational rank of `x`
* `cnames`: vector of column names
"""
mutable struct FeMat{T,S<:AbstractMatrix}
    x::S
    wtx::S
    piv::Vector{Int}
    rank::Int
    cnames::Vector{String}
end

function FeMat(X::AbstractMatrix, cnms)
    T = eltype(X)
    cf = cholesky!(Symmetric(X'X, :U), Val(true), tol = -one(T))
    r = cf.rank
    piv = cf.piv
    X = X[:, piv[1:r]]
    FeMat{T,typeof(X)}(X, X, piv, r, cnms)
end
#=
function FeMat(y::Vector{T}) where {T}
    m = reshape(y, (:, 1))
    FeMat{T,Matrix{T}}(m, m, [1], Int(!all(iszero, y)), [""])
end
=#
function reweight!(A::FeMat{T}, sqrtwts::Vector{T}) where {T}
    if !isempty(sqrtwts)
        if (A.x === A.wtx)
            A.wtx = similar(A.x)
        end
        mul!(A.wtx, Diagonal(sqrtwts), A.x)
    end
    A
end

Base.adjoint(A::FeMat) = Adjoint(A)

Base.eltype(A::FeMat{T}) where {T} = T

Base.length(A::FeMat) = length(A.wtx)

Base.size(A::FeMat) = size(A.wtx)

Base.size(A::Adjoint{T,<:FeMat{T}}) where {T} = reverse(size(A.parent))

Base.size(A::FeMat, i) = size(A.wtx, i)

Base.copyto!(A::FeMat{T}, src::AbstractVecOrMat{T}) where {T} = copyto!(A.x, src)

*(A::Adjoint{T,<:FeMat{T}}, B::FeMat{T}) where {T} = A.parent.wtx'B.wtx

LinearAlgebra.mul!(R::AbstractMatrix{T}, A::FeMat{T}, B::FeMat{T}) where {T} =
    mul!(R, A.wtx, B.wtx)

LinearAlgebra.mul!(R::StridedVecOrMat{T}, A::FeMat{T}, B::StridedVecOrMat{T}) where {T} =
    mul!(R, A.x, B)

LinearAlgebra.mul!(C, A::Adjoint{T,<:FeMat{T}}, B::FeMat{T}) where {T} =
    mul!(C, A.parent.wtx', B.wtx)
