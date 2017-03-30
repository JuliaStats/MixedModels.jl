nθ(A::UniformScaling) = 1
nθ(A::MaskedMatrix) = length(A.mask)

"""
    getθ!{T}(v::AbstractVector{T}, A::MaskedMatrix{T})

Overwrite `v` with the elements of the blocks in the lower triangle of `A` (column-major ordering)
"""
function getθ!{T<:AbstractFloat}(v::StridedVector{T}, A::MaskedMatrix{T})
    @argcheck length(v) == length(A.mask) DimensionMismatch
    mask = A.mask
    m = A.m
    for i in eachindex(mask)
        v[i] = m[mask[i]]
    end
    v
end

function getθ!{T}(v::StridedVector{T}, A::UniformScaling{T})
    @argcheck length(v) == 1 DimensionMismatch
    v[1] = A.λ
    v
end

"""
    getθ(A::MaskedMatrix{T})

Return a vector of the elements of the lower triangle blocks in `A` (column-major ordering)
"""
getθ(A::UniformScaling) = [A.λ]
getθ(A::MaskedMatrix) = A.m[A.mask]
getθ(A::AbstractVector) = mapreduce(getθ, vcat, A)

"""
    lowerbd{T}(A::LowerTriangular{T})

Return the vector of lower bounds on the parameters, `θ`.

These are the elements in the lower triangle in column-major ordering.
Diagonals have a lower bound of `0`.  Off-diagonals have a lower-bound of `-Inf`.
"""
lowerbd(v::AbstractVector) = mapreduce(lowerbd, vcat, v)
lowerbd{T}(A::UniformScaling{T}) = zeros(T, (1,))
lowerbd{T}(A::MaskedMatrix{T}) = T[x ∈ diagind(A.m) ? zero(T) : convert(T, -Inf) for x in A.mask]

"""
    LT(A)

Create a uniform scaling, lower triangular, matrix compatible with the term `A`
and initialized to the identity.
"""
LT{T}(A::ScalarReMat{T}, indep) = UniformScaling(one(T))
function LT{T}(A::VectorReMat{T}, indep)
    n = size(A.z, 1)
    blkend = get(indep, A.fnm, [n])
    if  any(diff(blkend) .< 1) || blkend[1] < 1 || blkend[end] ≠ n
        throw(ArgumentError("indep[$(A.fnm)] = $(indep[A.fnm]) is malformed"))
    end
    inds = reshape(1:abs2(n), (n, n))
    mask = Int[]
    offset = 0
    for b in blkend
        k = b - offset
        for j in 1:k, i in j:k
            push!(mask, inds[offset + i, offset + j])
        end
        offset = b
    end
    MaskedMatrix(eye(T, n), mask)
end

function Λvec(trms, indep)
    Union{UniformScaling, MaskedMatrix}[LT(x, indep) for x in trms]
end

function setθ!{T}(A::MaskedMatrix{T}, v::AbstractVector{T})
    copy!(view(A.m, A.mask), v)
    A
end
