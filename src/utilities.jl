"""
    allequal(x::Array)
    allequal(x::Tuple)
Return the equality of all elements of the array
"""
function allequal(x::Array; comparison=isequal)::Bool
    # the ref is necessary in case the elements of x are themselves arrays
    all(comparison.(x,  Ref(first(x))))
end

allequal(x::Vector{Bool})::Bool = !any(x) || all(x)

allequal(x::NTuple{N,Bool}) where {N} = !any(x) || all(x)

function allequal(x::Tuple; comparison=isequal)::Bool
    all(comparison.(x,  Ref(first(x))))
end

function allequal(x...; comparison=isequal)::Bool
    all(comparison.(x,  Ref(first(x))))
end

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

function rownormalize!(A::AbstractMatrix)
    for r in eachrow(A)
        # all zeros arise in zerocorr situations
        if !iszero(r)
            normalize!(r)
        end
    end
    A
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
    PCA{T<:AbstractFloat}

Principal Components Analysis

## Fields

* `covcorr` covariance or correlation matrix
* `sv` singular value decomposition
* `rnames` rownames of the original matrix
* `corr` is this a correlation matrix?
"""
struct PCA{T<:AbstractFloat}
    covcor::Symmetric{T,Matrix{T}}
    sv::SVD{T,T,Matrix{T}}
    rnames::Union{Vector{String}, Missing}
    corr::Bool
end

"""
    PCA(::AbstractMatrix; corr::Bool=true)
    PCA(::ReMat; corr::Bool=true)
    PCA(::LinearMixedModel; corr::Bool=true)

Constructs a [`MixedModels.PCA`](@ref]) object from a covariance matrix.

For `LinearMixedModel`, a named tuple of PCA on each of the random-effects terms is returned.

If `corr=true`, then the covariance is first standardized to the correlation scale.
"""

function PCA(covfac::AbstractMatrix, rnames=missing; corr::Bool=true)
    covf = corr ? rownormalize!(copy(covfac)) : copy(covfac)
    PCA(Symmetric(covf*covf', :L), svd(covf), rnames, corr)
end

function Base.getproperty(pca::PCA, s::Symbol)
    if s == :cumvar
        cumvv = cumsum(abs2.(pca.sv.S))
        cumvv ./ last(cumvv)
    elseif s == :loadings
        pca.sv.U
    else
        getfield(pca, s)
    end
end

Base.propertynames(pca::PCA, private::Bool = false) = (
    :covcor,
    :sv,
    :corr,
    :cumvar,
    :loadings,
#    :rotation,
)

Base.show(pca::PCA;
        ndigitsmat=2, ndigitsvec=2, ndigitscum=4,
        covcor=true, loadings=true, variances=false, stddevs=false) =
        Base.show(Base.stdout, pca,
                    ndigitsmat=ndigitsmat,
                    ndigitsvec=ndigitsvec,
                    ndigitscum=ndigitscum,
                    covcor=covcor,
                    loadings=loadings,
                    variances=variances,
                    stddevs=stddevs)

function Base.show(io::IO, pca::PCA;
        ndigitsmat=2, ndigitsvec=2, ndigitscum=4,
        covcor=true, loadings=true, variances=false, stddevs=false)
    println(io)
    if covcor
        println(io,
                "Principal components based on ",
                pca.corr ? "correlation" : "(relative) covariance",
                " matrix")
        # only display the lower triangle of symmetric matrix
        if pca.rnames !== missing
            n = length(pca.rnames)
            cv = string.(round.(pca.covcor, digits=ndigitsmat))
            dotpad = lpad(".", div(maximum(length, cv),2))
            for i = 1:n, j = (i+1):n
                cv[i, j] = dotpad
            end
            neg = startswith.(cv, "-")
            if any(neg)
                cv[.!neg] .= " ".* cv[.!neg]
            end
            # this hurts type stability,
            # but this show method shouldn't be a bottleneck
            printmat = Text.([pca.rnames cv])
        else
            # if there are no names, then we cheat and use the print method
            # for LowerTriangular, which automatically covers the . in the
            # upper triangle
            printmat = round.(LowerTriangular(pca.covcor), digits=ndigitsmat)
        end

        Base.print_matrix(io, printmat)
        println(io)
    end
    if stddevs
        println(io, "\nStandard deviations:")
        sv = pca.sv
        show(io, round.(sv.S, digits=ndigitsvec))
        println(io)
    end
    if variances
        println(io, "\nVariances:")
        vv = abs2.(sv.S)
        show(io, round.(vv, digits=ndigitsvec))
        println(io)
    end
    println(io, "\nNormalized cumulative variances:")
    show(io, round.(pca.cumvar, digits=ndigitscum))
    println(io)
    if loadings
        println(io, "\nComponent loadings")
        printmat = round.(pca.loadings, digits=ndigitsmat)

        if pca.rnames !== missing
            pclabs = [Text(""); Text.( "PC$i" for i in 1:length(pca.rnames))]
            pclabs = reshape(pclabs, 1, :)
            # this hurts type stability,
            # but this show method shouldn't be a bottleneck
            printmat = [pclabs; Text.(pca.rnames) printmat]
        end

        Base.print_matrix(io, printmat)
    end

    nothing
end
