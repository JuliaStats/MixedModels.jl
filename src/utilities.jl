"""
    average(a::T, b::T) where {T<:AbstractFloat}

Return the average of `a` and `b`
"""
average(a::T, b::T) where {T<:AbstractFloat} = (a + b) / 2

"""
    cpad(s::AbstractString, n::Integer)

Return a string of length `n` containing `s` in the center (more-or-less).
"""
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
    elseif nnz(A) / (m * n) ≤ threshold
        A
    else
        Array(A)
    end
end
densify(A::AbstractMatrix, threshold::Real = 0.3) = A

densify(A::SparseVector, threshold::Real = 0.3) = Vector(A)
densify(A::Diagonal{T,SparseVector}, threshold::Real = 0.3) where {T} =
    Diagonal(Vector(A.diag))

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

function Base.sum!(s::AbstractVector{T}, a::RaggedArray{T}) where {T}
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
    replicate(f::Function, n::Integer, use_threads=false)

Return a vector of the values of `n` calls to `f()` - used in simulations where the value of `f` is stochastic.

Note that if `f()` is not thread-safe or depends on a non thread-safe RNG,
    then you must set `use_threads=false`. Also note that ordering of replications
    is not guaranteed when `use_threads=true`, although the replications are not
    otherwise affected for thread-safe `f()`.
"""
function replicate(f::Function, n::Integer; use_threads=false)
    if use_threads
        # no macro version yet: https://github.com/timholy/ProgressMeter.jl/issues/143
        p = Progress(n)
        # get the type
        rr = f()
        next!(p)
        # pre-allocate
        results = [rr for _ in Base.OneTo(n)]
        Threads.@threads for idx = 2:n
            results[idx] = f()
            next!(p)
        end
    else
        results = @showprogress [f() for _ in Base.OneTo(n)]
    end
    results
end

"""
    dataset(nm)

Return the data frame of test data set named `nm`, which can be a `String` or `Symbol`
"""
function dataset(nm::AbstractString)
    path = joinpath(TestData, nm * ".feather")
    if !isfile(path)
        throw(ArgumentError(
            "Dataset \"$nm\" is not available.\nUse MixedModels.datasets() for available names."))
    end
    Feather.read(path)
end
dataset(nm::Symbol) = dataset(string(nm))

"""
    datasets()

Return a vector of names of the available test data sets
"""
datasets() = first.(Base.Filesystem.splitext.(filter(Base.Fix2(endswith, ".feather"), readdir(TestData))))
