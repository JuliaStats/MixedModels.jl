"""
densify(S::SparseMatrix, threshold=0.3)

Convert sparse `S` to `Diagonal` if `S` is diagonal or to `full(S)` if
the proportion of nonzeros exceeds `threshold`.
"""
function densify(A::SparseMatrixCSC, threshold::Real = 0.3)
    m, n = size(A)
    if m == n && isdiag(A)  # convert diagonal sparse to Diagonal
        Diagonal(diag(A))
    elseif nnz(A)/(m * n) ≤ threshold
        A
    else
        Array(A)
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


function stddevcor!(σ::Vector{T}, ρ::Matrix{T}, scr::Matrix{T}, L::Cholesky{T}) where {T}
    length(σ) == (k = size(L, 2)) && size(ρ) == (k, k) && size(scr) == (k, k) || 
        throw(DimensionMismatch(""))
    if L.uplo == 'L'
        copyto!(scr, L.factors)
        for i in 1 : k
            σ[i] = σi = norm(view(scr, i, 1:i))
            for j in 1:i
                scr[i, j] /= σi
            end
        end
        mul!(ρ, LowerTriangular(scr), adjoint(LowerTriangular(scr)))
    elseif L.uplo == 'U'
        copyto!(scr, L.factors)
        for j in 1 : k
            σ[j] = σj = norm(view(scr, 1:j, j))
            for i in 1 : j
                scr[i, j] /= σj
            end
        end
        mul!(ρ, UpperTriangular(scr)', UpperTriangular(scr))
    else
        throw(ArgumentError("L.uplo should be 'L' or 'U'"))
    end
    σ, ρ
end

function stddevcor!(σ::Vector{T}, ρ::Matrix{T}, scr::Matrix{T}, L::LowerTriangular{T}) where {T}
    length(σ) == (k = size(L, 2)) && size(ρ) == (k, k) && size(scr) == (k, k) || 
        throw(DimensionMismatch(""))
    copyto!(scr, L)
    for i in 1:k
        σ[i] = σi = norm(view(scr, i, 1:i))
        for j in 1:i
            scr[i, j] /= σi
        end
    end
    mul!(ρ, LowerTriangular(scr), adjoint(LowerTriangular(scr)))
    σ, ρ
end

function stddevcor(L::Cholesky{T}) where {T}
    k = size(L, 1)
    stddevcor!(Vector{T}(undef, k), Matrix{T}(undef, k, k), Matrix{T}(undef, k, k), L)
end
