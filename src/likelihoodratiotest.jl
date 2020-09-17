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
    likelihoodratiotest(m::MixedModel...)

Likeihood ratio test applied to a set of nested models.

Note that nesting of the models is not checked.  It is incumbent on the user 
to check this. This differs from [`StatsModels.lrtest`](@ref) as nesting in 
mixed models, especially in the random effects specification, may be non obvious.

This functionality may be deprecated in the future in favor of [`StatsModels.lrtest`](@ref).
"""
function likelihoodratiotest(m::MixedModel...)
    _iscomparable(m...) || throw(
            ArgumentError("""Models are not comparable: are the objectives, data
                             and, where approriate, the link and family the same?
            """))
    
    m = collect(m)   # change the tuple to an array
    dofs = dof.(m)
    formulas = String.(Symbol.(getproperty.(m,:formula)))
    ord = sortperm(dofs)
    dofs = dofs[ord]
    formulas = formulas[ord]
    devs = objective.(m)[ord]
    dofdiffs = diff(dofs)
    devdiffs = .-(diff(devs))
    pvals = map(zip(dofdiffs, devdiffs)) do (dof, dev)
        if dev > 0
            ccdf(Chisq(dof), dev)
        else
            NaN
        end
    end

    LikelihoodRatioTest(
        formulas,
        (dof = dofs, deviance = devs),
        (dofdiff = dofdiffs, deviancediff = devdiffs, pvalues = pvals)
    )
end

function Base.show(io::IO, lrt::LikelihoodRatioTest; digits=2)
    println(io, "Model Formulae")

    for (i, f) in enumerate(lrt.formulas)
        println(io, "$i: $f")
    end

    # the following was adapted from StatsModels#162
    # from nalimilan
    Δdf = lrt.tests.dofdiff
    Δdev = lrt.tests.deviancediff

    nc = 6
    nr = length(lrt.formulas)
    outrows = Matrix{String}(undef, nr+1, nc)

    outrows[1, :] = ["",
                    "model-dof",
                    "deviance",
                    "χ²",
                    "χ²-dof",
                    "P(>χ²)"] # colnms


    outrows[2, :] = ["[1]",
                    string(lrt.dof[1]),
                    Ryu.writefixed(lrt.deviance[1], 4),
                    " "," ", " "]

    for i in 2:nr
        outrows[i+1, :] = ["[$i]",
                           string(lrt.dof[i]),
                           Ryu.writefixed(lrt.deviance[i], 4),
                           Ryu.writefixed(Δdev[i-1], 4),
                           string(Δdf[i-1]),
                           string(StatsBase.PValue(lrt.pvalues[i-1]))]
    end
    colwidths = length.(outrows)
    max_colwidths = [maximum(view(colwidths, :, i)) for i in 1:nc]
    totwidth = sum(max_colwidths) + 2*5

    println(io, '─'^totwidth)

    for r in 1:nr+1
        for c in 1:nc
            cur_cell = outrows[r, c]
            cur_cell_len = length(cur_cell)

            padding = " "^(max_colwidths[c]-cur_cell_len)
            if c > 1
                padding = "  "*padding
            end

            print(io, padding)
            print(io, cur_cell)
        end
        print(io, "\n")
        r == 1 && println(io, '─'^totwidth)
    end
    print(io, '─'^totwidth)

    nothing
end

function _iscomparable(m::LinearMixedModel...)
    allequal(getproperty.(getproperty.(m,:optsum),:REML)) ||
        throw(ArgumentError("Models must all be fit with the same objective (i.e. all ML or all REML)"))
    
    if any(getproperty.(getproperty.(m,:optsum),:REML))
        allequal(coefnames.(m))  ||
                throw(ArgumentError("Likelihood-ratio tests for REML-fitted models are only valid when the fixed-effects specifications are identical"))
    end

    allequal(nobs.(m)) || 
        throw(ArgumentError("Models must have the same number of observations"))
    
    true
end

function _iscomparable(m::GeneralizedLinearMixedModel...)
        # TODO: test that all models are fit with same fast/nAGQ option?
        glms = getproperty.(m,:resp);
        
        allequal(Distribution.(glms)) ||
            throw(ArgumentError("Models must be fit to the same distribution"))
                
        allequal(string.(Link.(glms))) || 
            throw(ArgumentError("Models must have the same link function"))

        allequal(nobs.(m)) || 
            throw(ArgumentError("Models must have the same number of observations"))
        
        true
end

"""
    isnested(m1::MixedModel, m2::MixedModel; atol::Real=0.0)
Indicate whether model `m1` is nested in model `m2`, i.e. whether
`m1` can be obtained by constraining some parameters in `m2`.
Both models must have been fitted on the same data. This check
is conservative for `MixedModel`s and may reject nested models with different
parameterizations as being non nested. 
"""
function StatsModels.isnested(m1::MixedModel, m2::MixedModel; atol::Real=0.0)
    try
        _iscomparable(m1, m2)    
    catch e
        @error e.msg
        false
    end || return false
    
    # check that the nested fixef are a subset of the outer
    all(in.(coefnames(m1),  Ref(coefnames(m2)))) || return false
    

    # check that the same grouping vars occur in the outer model
    grpng1 = getproperty.(getproperty.(m1.reterms, :trm), :sym)
    grpng2 = getproperty.(getproperty.(m2.reterms, :trm), :sym)

    all(in.(grpng1, Ref(grpng2))) || return false

    # check that every intercept/slope for a grouping var occurs in the 
    # same grouping
    re1 = Dict(re.trm.sym => re.cnames for re in m1.reterms)
    re2 = Dict(re.trm.sym => re.cnames for re in m2.reterms)

    all(all(in.(val, Ref(re2[key]))) for (key, val) in re1) || return false

    true
end
