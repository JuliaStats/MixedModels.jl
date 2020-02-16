"""
    LikelihoodRatioTest

Results of MixedModels.likelihoodratiotest

## Fields
* `formulas`: Vector of model formulae
* `models`: NamedTuple of the `dof` and `deviance` of the models
* `tests`: NamedTuple of the sequential `dofdiff`, `deviancediff`, and resulting `pvalues`

## Properties
* `deviance`
* `pvalues`

"""
struct LikelihoodRatioTest
    formulas::AbstractVector{String}
    models::NamedTuple{(:dof,:deviance)}
    tests::NamedTuple{(:dofdiff,:deviancediff,:pvalues)}
end

Base.propertynames(lrt::LikelihoodRatioTest, private = false) = (
    :deviance,
    :formulas,
    :models,
    :pvalues,
    :tests,
)

function Base.getproperty(lrt::LikelihoodRatioTest, s::Symbol)
    if s == :dof
        lrt.models.dof
    elseif s == :deviance
        lrt.models.deviance
    elseif s == :pvalues
        lrt.tests.pvalues
    elseif s == :formulae
        lrt.formulas
    else
        getfield(lrt, s)
    end
end

# backward syntactic but not type compatibility
Base.getindex(lrt::LikelihoodRatioTest, s::Symbol) = getfield(lrt,s)

"""
    likelihoodratiotest(m::LinearMixedModel...)

Likeihood ratio test applied to a set of nested models.

Note that nesting of the models is not checked.  It is incumbent on the user to check this.
"""
function likelihoodratiotest(m::LinearMixedModel...)
    if any(getproperty.(getproperty.(m,:optsum),:REML))
        reduce(==,coefnames.(m))  ||
                throw(ArgumentError("Likelihood-ratio tests for REML-fitted models are only valid when the fixed-effects specifications are identical"))
    end
    m = collect(m)   # change the tuple to an array
    dofs = dof.(m)
    formulas = String.(Symbol.(getproperty.(m,:formula)))
    ord = sortperm(dofs)
    dofs = dofs[ord]
    formulas = formulas[ord]
    devs = objective.(m)[ord]
    dofdiffs = diff(dofs)
    devdiffs = .-(diff(devs))
    pvals = ccdf.(Chisq.(dofdiffs), devdiffs)

    LikelihoodRatioTest(
        formulas,
        (dof = dofs, deviance = devs),
        (dofdiff = dofdiffs, deviancediff = devdiffs, pvalues = pvals)
    )
end

function _array_union_nothing(arr::Array{T}) where T
    Array{Union{T,Nothing}}(arr)
end

function _prepend_0(arr::Array{T}) where T
    pushfirst!(copy(arr), -zero(T))
end

function Base.show(io::IO, lrt::LikelihoodRatioTest; digits=2)
    println(io, "Model Formulae")

    for (i, f) in enumerate(lrt.formulas)
        println("$i: $f")
    end
    cols = hcat(lrt.models.dof, lrt.models.deviance,
                _prepend_0(lrt.tests.deviancediff),
                _prepend_0(lrt.tests.dofdiff),
                _prepend_0(lrt.tests.pvalues))

    ct = CoefTable(
        cols, # cols
        ["model-dof", "deviance", "χ²", "χ²-dof", "P(>χ²)"], # colnms
        string.(1:length(lrt.formulas)), # rownms
        5, # pvalcol
        3 # teststatcol
    )
    show(io, ct)

    nothing
end
