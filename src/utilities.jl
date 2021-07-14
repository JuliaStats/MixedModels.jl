"""
    _abstractify_grouping(f::FormulaTerm)

Remove concrete levels associated with a schematized FormulaTerm.

Returns the formula with the grouping variables made abstract again
and a Dictionary of `Grouping()` contrasts.
"""
function _abstractify_grouping(f::FormulaTerm)
    fe = filter(x -> !isa(x, AbstractReTerm), f.rhs)
    re = filter(x -> isa(x, AbstractReTerm), f.rhs)
    contr = Dict{Symbol, AbstractContrasts}()
    re = map(re) do trm
        if trm.rhs isa InteractionTerm
            rhs = mapreduce(&, trm.rhs.terms) do tt
                contr[tt.sym] = Grouping()
                return Term(tt.sym)
            end
        else
            contr[trm.rhs.sym] = Grouping()
            rhs = Term(trm.rhs.sym)
        end
        return trm.lhs | rhs
    end
    return (f.lhs ~ sum(fe) + sum(re)), contr
end

"""
    isconstant(x::Array)
    isconstant(x::Tuple)

Are all elements of the iterator the same?  That is, is it constant?
"""
function isconstant(x; comparison=isequal)::Bool
    # the ref is necessary in case the elements of x are themselves arrays
    isempty(x) || all(ismissing, x) || coalesce(all(comparison.(x,  Ref(first(x)))), false)
end

isconstant(x::Vector{Bool})::Bool = !any(x) || all(x)

isconstant(x...; comparison=isequal) = isconstant(x; comparison=comparison)

"""
    average(a::T, b::T) where {T<:AbstractFloat}

Return the average of `a` and `b`
"""
average(a::T, b::T) where {T<:AbstractFloat} = (a + b) / T(2)

"""
    cpad(s::AbstractString, n::Integer)

Return a string of length `n` containing `s` in the center (more-or-less).
"""
cpad(s::String, n::Integer) = rpad(lpad(s, (n + textwidth(s)) >> 1), n)

"""
densify(S::SparseMatrix, threshold=0.1)

Convert sparse `S` to `Diagonal` if `S` is diagonal or to `Array(S)` if
the proportion of nonzeros exceeds `threshold`.
"""
function densify(A::SparseMatrixCSC, threshold::Real = 0.1)
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
densify(A::AbstractMatrix, threshold::Real = 0.1) = A

densify(A::SparseVector, threshold::Real = 0.1) = Vector(A)
densify(A::Diagonal{T,SparseVector{T,Ti}}, threshold::Real = 0.1) where {T,Ti} =
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

function rownormalize(A::AbstractMatrix)
    A = copy(A)
    for r in eachrow(A)
        # all zeros arise in zerocorr situations
        if !iszero(r)
            normalize!(r)
        end
    end
    A
end

function rownormalize(A::LowerTriangular{T, Diagonal{T, Vector{T}}}) where T
    one(T) * I(size(A,1))
end

# from the ProgressMeter docs
_is_logging(io) = isa(io, Base.TTY) == false || (get(ENV, "CI", nothing) == "true")

"""
    replicate(f::Function, n::Integer; use_threads=false)

Return a vector of the values of `n` calls to `f()` - used in simulations where the value of `f` is stochastic.

`hide_progress` can be used to disable the progress bar. Note that the progress
bar is automatically disabled for non-interactive (i.e. logging) contexts.

!!! warning
    If `f()` is not thread-safe or depends on a non thread-safe RNG,
    then you must set `use_threads=false`. Also note that ordering of replications
    is not guaranteed when `use_threads=true`, although the replications are not
    otherwise affected for thread-safe `f()`.
"""
function replicate(f::Function, n::Integer;
                   use_threads=false, hide_progress=false)
    # no macro version yet: https://github.com/timholy/ProgressMeter.jl/issues/143
    # and we want some advanced options
    p = Progress(n; output=Base.stderr,
                 enabled=!hide_progress && !_is_logging(stderr))
    # get the type
    rr = f()
    next!(p)
    # pre-allocate
    results = [rr for _ in Base.OneTo(n)]
    if use_threads
        Threads.@threads for idx = 2:n
            results[idx] = f()
            next!(p)
        end
    else
        for idx = 2:n
            results[idx] = f()
            next!(p)
        end
    end
    results
end

"""
    sdcorr(A::AbstractMatrix{T}) where {T}

Transform a square matrix `A` with positive diagonals into an `NTuple{size(A,1), T}` of
standard deviations and a tuple of correlations.

`A` is assumed to be symmetric and only the lower triangle is used.  The order of the
correlations is row-major ordering of the lower triangle (or, equivalently, column-major
in the upper triangle).
"""
function sdcorr(A::AbstractMatrix{T}) where {T}
    m,n = size(A)
    m == n || throw(ArgumentError("matrix A must be square"))
    indpairs = checkindprsk(m)
    rtdiag = sqrt.(NTuple{m,T}(diag(A)))
    (
        rtdiag,
        ntuple(kchoose2(m)) do k
            i,j = indpairs[k]
            A[i,j]/(rtdiag[i] * rtdiag[j])
        end,
    )
end
