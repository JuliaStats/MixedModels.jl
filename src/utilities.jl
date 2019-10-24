average(a::T, b::T) where {T<:AbstractFloat} = (a + b) / 2

cpad(s::String, n::Integer) = rpad(lpad(s, (n + textwidth(s)) >> 1), n)

"""
densify(S::SparseMatrix, threshold=0.25)

Convert sparse `S` to `Diagonal` if `S` is diagonal or to `Array(S)` if
the proportion of nonzeros exceeds `threshold`.
"""
function densify(A::SparseMatrixCSC, threshold::Real = 0.25)
    m, n = size(A)
    if m == n && isdiag(A)  # convert diagonal sparse to Diagonal
        # the diagonal is always dense (otherwise rank deficit)
        # so make sure it's stored as such
        Diagonal(Vector(diag(A)))
    elseif nnz(A)/(m * n) ≤ threshold
        A
    else
        Array(A)
    end
end
densify(A::AbstractMatrix, threshold::Real = 0.3) = A

densify(A::SparseVector, threshold::Real = 0.3)  = Vector(A)
densify(A::Diagonal{T, SparseVector}, threshold::Real = 0.3) where T = Diagonal(Vector(A.diag))

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
    vars ./ last(vars)
end

"""
    ltriindprs

A row-major order `Vector{NTuple{2,Int}}` of indices in the strict lower triangle.
"""
const ltriindprs = NTuple{2,Int}[]

function checkindprsk(k::Integer)
    kchoose2 = (k * (k - 1)) >> 1
    if length(ltriindprs) < kchoose2
        sizehint!(empty!(ltriindprs), kchoose2)
        for i in 1:k, j in 1:(i-1)
            push!(ltriindprs, (i,j))
        end
    end
    ltriindprs
end

"""
    subscriptednames(nm, len)

Return a `Vector{String}` of `nm` with subscripts from `₁` to `len`
"""
function subscriptednames(nm, len)
    nd = ndigits(len)
    nd == 1 ?
        [string(nm, '₀' + j) for j in 1:len] :
        [string(nm, lpad(string(j), nd, '0')) for j in 1:len]
end
#=
updatedevresid!(r::GLM.GlmResp, η::AbstractVector) = updateμ!(r, η)

fastlogitdevres(η, y) = 2log1p(exp(iszero(y) ? η : -η))

function updatedevresid!(r::GLM.GlmResp{V,<:Bernoulli,LogitLink}, η::V) where V<:AbstractVector{<:AbstractFloat}
    map!(fastlogitdevres, r.devresid, η, r.y)
    r
end
=#
