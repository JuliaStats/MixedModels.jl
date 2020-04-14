"""
    allequal(x::Array)
    allequal(x::Tuple)
Return the equality of all elements of the array
"""
function allequal(x::Array; comparison=isequal)::Bool
    all(comparison.(first(x),  x))
end

allequal(x::Vector{Bool})::Bool = !any(x) || all(x)

allequal(x::NTuple{N,Bool}) where {N} = !any(x) || all(x)

function allequal(x::Tuple; comparison=isequal)::Bool
    all(comparison.(first(x),  x))
end

function allequal(x...; comparison=isequal)::Bool
    all(comparison.(first(x),  x))
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
densify(A::Diagonal{T,SparseVector{T,Ti}}, threshold::Real = 0.3) where {T,Ti} =
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

"""
    replicate(f::Function, n::Integer; use_threads=false)

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

cacheddatasets = Dict{String,Any}()
"""
    dataset(nm)

Return the data frame of test data set named `nm`, which can be a `String` or `Symbol`
"""
function dataset(nm::AbstractString)
    get!(cacheddatasets, nm) do
        path = joinpath(TestData, nm * ".feather")
        if !isfile(path)
            throw(ArgumentError(
                "Dataset \"$nm\" is not available.\nUse MixedModels.datasets() for available names."))
        end
        Feather.read(path)
    end
end
dataset(nm::Symbol) = dataset(string(nm))

"""
    datasets()

Return a vector of names of the available test data sets
"""
datasets() = first.(Base.Filesystem.splitext.(filter(Base.Fix2(endswith, ".feather"), readdir(TestData))))


"""
    PCA{T<:AbstractFloat}

Principal Components Analysis

## Fields

* `covcorr` covariance or correlation matrix
* `sv` singular value decomposition
* `corr` is this a correlation matrix
"""
struct PCA{T<:AbstractFloat}
    covcor::Symmetric{T,Matrix{T}}
    sv::SVD{T,T,Matrix{T}}
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

function PCA(covfac::AbstractMatrix; corr::Bool=true)
    covf = corr ? rownormalize!(copy(covfac)) : copy(covfac)
    PCA(Symmetric(covf*covf', :L), svd(covf), corr)
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

Base.propertynames(pca::PCA, private = false) = (
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
        Base.print_matrix(io, round.(LowerTriangular(pca.covcor), digits=ndigitsmat))
        println(io)
    end
    if stddevs
        println(io, "Standard deviations:")
        sv = pca.sv
        show(io, round.(sv.S, digits=ndigitsvec))
        println(io)
    end
    if variances
        println(io, "Variances:")
        vv = abs2.(sv.S)
        show(io, round.(vv, digits=ndigitsvec))
        println(io)
    end
    println(io, "Normalized cumulative variances:")
    show(io, round.(pca.cumvar, digits=ndigitscum))
    println(io)
    if loadings
        println(io, "Component loadings")
        Base.print_matrix(io, round.(pca.loadings, digits=ndigitsmat))
    end

    nothing
end
