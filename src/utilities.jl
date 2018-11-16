"""
densify(S::SparseMatrix, threshold=0.3)

Convert sparse `S` to `Diagonal` if `S` is diagonal or to `full(S)` if
the proportion of nonzeros exceeds `threshold`.
"""
function densify(A::SparseMatrixCSC, threshold::Real = 0.3)
S = sparse(A)
m, n = size(S)
if m == n && isdiag(S)  # convert diagonal sparse to Diagonal
    Diagonal(diag(S))
elseif nnz(S)/(S.m * S.n) ≤ threshold
    A
else
    Array(S)
end
end
densify(A::AbstractMatrix, threshold::Real = 0.3) = A

"""
    RaggedArray{T,I}

A "ragged" array structure consisting of values and indices

# Fields
- `vals`: a `Vector{T}` containing the values
- `inds`: a `Vector{I}` containing the indices

For this application a `RaggedArray` is used only in its `sum!` method.
"""
struct RaggedArray{T,I}
    vals::Vector{T}
    inds::Vector{I}
end
function Base.sum!(s::AbstractVector{T}, a::RaggedArray{T}) where T
    for (v, i) in zip(a.vals, a.inds)
        s[i] += v
    end
    s
end

"""
    normalized_variance_cumsum(A::AbstractMatrix)

Return the cumulative sum of the squared singular values of `A` normalized to sum to 1
"""
function normalized_variance_cumsum(A::AbstractMatrix)
    vars = cumsum(abs2.(svdvals(A)))
    vars ./ vars[end]
end
