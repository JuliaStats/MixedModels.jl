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
    covcor::Symmetric{T,<:AbstractMatrix{T}}
    sv::SVD{T,T,<:AbstractMatrix{T}}
    rnames::Union{Vector{String},Missing}
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
    covf = corr ? rownormalize(covfac) : covfac
    return PCA(Symmetric(covf * covf', :L), svd(covf), rnames, corr)
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

function Base.propertynames(pca::PCA, private::Bool=false)
    return (
        :covcor,
        :sv,
        :corr,
        :cumvar,
        :loadings,
        #    :rotation,
    )
end

Base.show(io::IO, pca::PCA; kwargs...) = Base.show(io, MIME"text/plain"(), pca; kwargs...)

function Base.show(
    io::IO,
    ::MIME"text/plain",
    pca::PCA;
    ndigitsmat=2,
    ndigitsvec=2,
    ndigitscum=4,
    covcor=true,
    loadings=true,
    variances=false,
    stddevs=false,
)
    println(io)
    if covcor
        println(
            io,
            "Principal components based on ",
            pca.corr ? "correlation" : "(relative) covariance",
            " matrix",
        )
        # only display the lower triangle of symmetric matrix
        if pca.rnames !== missing
            n = length(pca.rnames)
            cv = string.(round.(pca.covcor, digits=ndigitsmat))
            dotpad = lpad(".", div(maximum(length, cv), 2))
            for i in 1:n, j in (i + 1):n
                cv[i, j] = dotpad
            end
            neg = startswith.(cv, "-")
            if any(neg)
                cv[.!neg] .= " " .* cv[.!neg]
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
            pclabs = [Text(""); Text.("PC$i" for i in 1:length(pca.rnames))]
            pclabs = reshape(pclabs, 1, :)
            # this hurts type stability,
            # but this show method shouldn't be a bottleneck
            printmat = [pclabs; Text.(pca.rnames) Matrix(printmat)]
        end

        Base.print_matrix(io, printmat)
    end

    return nothing
end
