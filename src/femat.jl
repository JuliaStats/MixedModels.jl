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
    if size(X,2) > 0
        ch = statscholesky(Symmetric(X'X))
        pivot = ch.piv
        rank = ch.rank
        Xp = all(pivot .== 1:size(X, 2)) ? X : X[:, ch.piv]
        if rank < length(pivot)
            @warn "fixed-effects matrix is rank deficient"
        end
    else
        # although it doesn't take long for an empty matrix,
        # we can still skip the Cholesky step, which gets the rank
        # wrong anyway
        pivot = Int[]
        Xp = X
        rank = 0
    end
    # ch.rank is wrong for empty FE
    FeMat{T,typeof(X)}(Xp, Xp, pivot, rank, cnms[pivot])
end

function reweight!(A::FeMat{T}, sqrtwts::Vector{T}) where {T}
    if !isempty(sqrtwts)
        if A.x === A.wtx
            A.wtx = similar(A.x)
        end
        mul!(A.wtx, Diagonal(sqrtwts), A.x)
    end
    A
end

Base.adjoint(A::FeMat) = Adjoint(A)

Base.eltype(A::FeMat{T}) where {T} = T

fullrankwtx(A::FeMat) = rank(A) == size(A, 2) ? A.wtx : A.wtx[:, 1:rank(A)]

Base.length(A::FeMat) = length(A.wtx)

LinearAlgebra.rank(A::FeMat) = A.rank

Base.size(A::FeMat) = size(A.wtx)

Base.size(A::Adjoint{T,<:FeMat{T}}) where {T} = reverse(size(A.parent))

Base.size(A::FeMat, i) = size(A.wtx, i)

Base.copyto!(A::FeMat{T}, src::AbstractVecOrMat{T}) where {T} = copyto!(A.x, src)

*(adjA::Adjoint{T,<:FeMat{T}}, B::FeMat{T}) where {T} =
    fullrankwtx(adjA.parent)' * fullrankwtx(B)

LinearAlgebra.mul!(R::StridedVecOrMat{T}, A::FeMat{T}, B::StridedVecOrMat{T}) where {T} =
    mul!(R, A.x, B)

LinearAlgebra.mul!(C, adjA::Adjoint{T,<:FeMat{T}}, B::FeMat{T}) where {T} =
    mul!(C, fullrankwtx(adjA.parent)', fullrankwtx(B))

"""
    isfullrank(A::FeMat)

Does `A` have full column rank?
"""
isfullrank(A::FeMat) = A.rank == length(A.piv)

nθ(A::FeMat) = 0
