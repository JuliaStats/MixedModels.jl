"""
    HeteroBlkdMatrix

A matrix composed of heterogenous blocks.  Blocks can be sparse, dense or
diagonal.
"""

immutable HeteroBlkdMatrix <: AbstractMatrix{AbstractMatrix}
    blocks::Matrix{AbstractMatrix}
end
Base.size(A::HeteroBlkdMatrix) = size(A.blocks)
Base.getindex(A::HeteroBlkdMatrix, i::Int) = A.blocks[i]
Base.setindex!(A::HeteroBlkdMatrix, X, i::Integer) = setindex!(A.blocks, X, i)
@compat Base.IndexStyle(A::HeteroBlkdMatrix) = IndexLinear()

immutable Identity{T<:AbstractFloat} end

"""
    T<:AbstractFloat}

A `LowerTriangular{T, Matrix{T}}` and an integer vector, `mask`, of the potential non-zero elements.

In linear algebra operations an object `A` of this type acts like `I ⊗ A`,
for a suitably sized `I`.  These are the pattern matrices for blocks of `Λ`.
"""
immutable MaskedLowerTri{T<:AbstractFloat}
    m::LowerTriangular{T,Matrix{T}}
    mask::Vector{Int}
end
function MaskedLowerTri(v::Vector, T::DataType)
    n = sum(v)
    inds = reshape(1:abs2(n), (n, n))
    offset = 0
    mask = sizehint!(Int[], (n * (n + 1)) >> 1)
    for k in v
        for j in 1:k, i in j:k
            push!(mask, inds[offset + i, offset + j])
        end
        offset += k
    end
    MaskedLowerTri(LowerTriangular(eye(T, n)), mask)
end

=={T}(A::MaskedLowerTri{T}, B::MaskedLowerTri{T}) = A.m == B.m && A.mask == B.mask

"""
    LambdaTypes{T<:AbstractFloat}

Union of possible types in the `Λ` member of `[LinearMixedModel](@ref)`

These types are `Identity{T}`, `MaskedLowerTri{T}`, and `UniformScaling{T}`
"""
@compat const LambdaTypes{T} = Union{Identity{T}, MaskedLowerTri{T}, UniformScaling{T}}

if VERSION < v"0.6.0-pre.alpha"
    function cond{T}(J::UniformScaling{T})
        onereal = inv(one(real(J.λ)))
        return J.λ ≠ zero(T) ? onereal : oftype(onereal, Inf)
    end
end

cond(A::MaskedLowerTri) = cond(A.m)
cond{T}(I::Identity{T}) = one(T)

"""
    nθ(A::MaskedLowerTri)
    nθ(J::UniformScaling)
    nθ(J::Identity)

Return the number of free parameters in the Matrix
"""
function nθ end

nθ(A::MaskedLowerTri) = length(A.mask)
nθ(J::UniformScaling) = 1
nθ(J::Identity) = 0
nθ(v::Vector) = sum(nθ, v)

"""
    getθ!{T}(v::AbstractVector{T}, A::MaskedLowerTri{T})

Overwrite `v` with the elements of the blocks in the lower triangle of `A` (column-major ordering)
"""
function getθ!{T<:AbstractFloat}(v::StridedVector{T}, A::MaskedLowerTri{T})
    @argcheck length(v) == length(A.mask) DimensionMismatch
    mask = A.mask
    m = A.m.data
    @inbounds for i in eachindex(mask)
        v[i] = m[mask[i]]
    end
    v
end

function getθ!{T<:AbstractFloat}(v::StridedVector{T}, U::UniformScaling{T})
    @argcheck length(v) == 1 DimensionMismatch
    v[1] = U.λ
    v
end

function getθ!{T}(v::StridedVector{T}, I::Identity{T})
    @argcheck length(v) == 0 DimensionMisMatch
    v
end

"""
    getθ(A::MaskedLowerTri{T})

Return a vector of the elements of the lower triangle blocks in `A` (column-major ordering)
"""
getθ(A::MaskedLowerTri) = A.m.data[A.mask]
getθ(U::UniformScaling) = [U.λ]
getθ{T}(I::Identity{T}) = T[]
getθ(v::Vector) = mapreduce(getθ, vcat, v)

"""
    lowerbd{T}(A::MaskedLowerTri{T})
    lowerbd{T}(J::UniformScaling{T})
    lowerbd{T}(J::Identity{T})
    lowerbd{T}(v::Vector{LambdaTypes{T}})

Return the vector of lower bounds on the parameters, `θ`.

These are the elements in the lower triangle in column-major ordering.
Diagonals have a lower bound of `0`.  Off-diagonals have a lower-bound of `-Inf`.
"""
function lowerbd end

lowerbd(v::Vector) = mapreduce(lowerbd, vcat, v)
lowerbd{T}(A::MaskedLowerTri{T}) = T[x ∈ diagind(A.m.data) ? zero(T) : convert(T, -Inf) for x in A.mask]
lowerbd{T}(U::UniformScaling{T}) = zeros(T, 1)
lowerbd{T}(I::Identity{T}) = T[]

function setθ!{T}(A::MaskedLowerTri{T}, v::AbstractVector{T})
    @argcheck length(v) == length(A.mask) DimensionMismatch
    m = A.m.data
    mask = A.mask
    @inbounds for i in eachindex(mask)
        m[mask[i]] = v[i]
    end
    A
end

function A_mul_B!{T}(C::Matrix{T}, A::Matrix{T}, B::Identity{T})
    @argcheck size(C) == size(A) DimensionMismatch
    copy!(C, A)
end

A_mul_B!{T}(A::Identity{T}, B::AbstractVecOrMat{T}) = B

A_mul_B!{T}(C::Matrix{T}, A::Matrix{T}, B::UniformScaling{T}) = scale!(copy!(C, A), B.λ)

A_mul_B!{T}(A::UniformScaling{T}, B::AbstractVecOrMat{T}) = scale!(B, A.λ)

A_mul_B!{T}(A::AbstractVecOrMat{T}, B::UniformScaling{T}) = scale!(A, B.λ)

function A_mul_B!{T}(A::Diagonal{T}, B::UniformScaling{T})
    scale!(A.diag, B.λ)
    A
end

function A_mul_B!{T<:AbstractFloat}(A::Diagonal{LowerTriangular{T, Matrix{T}}},
    B::MaskedLowerTri{T})
    λ = LowerTriangular(B.m)
    for a in A.diag
        A_mul_B!(a.data, λ)
    end
    A
end

A_mul_B!{T}(A::AbstractVecOrMat{T}, J::Identity{T}) = A

function A_mul_B!{T<:AbstractFloat,S}(A::SparseMatrixCSC{T,S}, B::MaskedLowerTri{T})
    λ = B.m
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

function A_mul_B!{T<:AbstractFloat}(A::MaskedLowerTri{T}, B::StridedVector{T})
    λ = A.m
    k = size(λ, 1)
    A_mul_B!(λ, reshape(B, (k, div(length(B), k))))
    B
end

function A_mul_B!{T<:AbstractFloat}(A::Matrix{T}, B::MaskedLowerTri{T})
    λ = B.m
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

function A_mul_B!{T}(C::StridedVecOrMat{T}, A::UniformScaling{T}, B::StridedVecOrMat{T})
    @argcheck size(C) == size(B) DimensionMismatch
    broadcast!(*, C, A.λ, B)
end

function A_mul_B!{T}(C::StridedVecOrMat{T}, A::MaskedLowerTri{T}, B::StridedVecOrMat{T})
    @argcheck size(C) == size(B) DimensionMismatch
    m = size(C, 1)
    λ = A.m
    k = size(λ, 1)
    A_mul_B!(λ, reshape(copy!(C, B), (k, size(C, 2) * div(m, k))))
    C
end

function Ac_mul_B!{T}(A::UniformScaling{T}, B::Diagonal{T})
    scale!(B.diag, A.λ)
    B
end

Ac_mul_B!{T}(J::Identity{T}, A::AbstractVecOrMat{T}) = A

Ac_mul_B!{T}(A::UniformScaling{T}, B::AbstractArray{T}) = scale!(B, A.λ)

function Ac_mul_B!{T}(A::MaskedLowerTri{T}, B::Diagonal{LowerTriangular{T,Matrix{T}}})
    λ = A.m
    for b in B.diag
        Ac_mul_B!(λ, b.data)
    end
    B
end

function Ac_mul_B!{T}(A::MaskedLowerTri{T}, B::StridedVecOrMat{T})
    λ = A.m
    k = size(λ, 1)
    m, n = size(B, 1), size(B, 2)
    Ac_mul_B!(λ, reshape(B, (k, div(m, k) * n)))
    B
end

function Ac_mul_B!{T<:AbstractFloat,S}(A::MaskedLowerTri{T}, B::SparseMatrixCSC{T,S})
    λ = A.m
    k = size(λ, 2)
    nz = nonzeros(B)
    for j in 1:B.n
        bnz = view(nz, nzrange(B, j))
        mbj = reshape(bnz, (k, div(length(bnz), k)))
        Ac_mul_B!(mbj, λ, mbj)
    end
    B
end
