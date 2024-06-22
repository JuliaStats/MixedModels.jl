"""
    LikelihoodRatioTest

Results of MixedModels.likelihoodratiotest

## Fields
* `formulas`: Vector of model formulae
* `models`: NamedTuple of the `dof` and `deviance` of the models
* `tests`: NamedTuple of the sequential `dofdiff`, `deviancediff`,
           and resulting `pvalues`

## Properties
* `deviance` : note that this is actually -2 log likelihood for linear models
               (i.e. without subtracting the constant for a saturated model)
* `pvalues`

"""
struct LikelihoodRatioTest
    formulas::AbstractVector{String}
    models::NamedTuple{(:dof, :deviance)}
    tests::NamedTuple{(:dofdiff, :deviancediff, :pvalues)}
    linear::Bool
end

function Base.propertynames(lrt::LikelihoodRatioTest, private::Bool=false)
    return (:deviance, :formulas, :models, :pvalues, :tests)
end

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
Base.getindex(lrt::LikelihoodRatioTest, s::Symbol) = getfield(lrt, s)

"""
    likelihoodratiotest(m::MixedModel...)
    likelihoodratiotest(m0::LinearModel, m::MixedModel...)
    likelihoodratiotest(m0::GeneralizedLinearModel, m::MixedModel...)
    likelihoodratiotest(m0::TableRegressionModel{LinearModel}, m::MixedModel...)
    likelihoodratiotest(m0::TableRegressionModel{GeneralizedLinearModel}, m::MixedModel...)


Likeihood ratio test applied to a set of nested models.

!!! note
    The nesting of the models is not checked.  It is incumbent on the user
    to check this. This differs from `StatsModels.lrtest` as nesting in
    mixed models, especially in the random effects specification, may be non obvious.

!!! note
    For comparisons between mixed and non-mixed models, the deviance for the non-mixed
    model is taken to be -2 log likelihood, i.e. omitting the additive constant for the
    fully saturated model. This is in line with the computation of the deviance for mixed
    models.

This functionality may be deprecated in the future in favor of `StatsModels.lrtest`.
"""
function likelihoodratiotest(m::MixedModel...)
    _iscomparable(m...) ||
        throw(ArgumentError("""Models are not comparable: are the objectives, data
                               and, where appropriate, the link and family the same?
              """))

    m = collect(m)   # change the tuple to an array
    dofs = dof.(m)
    formulas = String.(Symbol.(getproperty.(m, :formula)))
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

    return LikelihoodRatioTest(
        formulas,
        (dof=dofs, deviance=devs),
        (dofdiff=dofdiffs, deviancediff=devdiffs, pvalues=pvals),
        first(m) isa LinearMixedModel,
    )
end

_formula(::Union{LinearModel,GeneralizedLinearModel}) = "NA"
function _formula(x::TableRegressionModel{<:Union{LinearModel,GeneralizedLinearModel}})
    return String(Symbol(x.mf.f))
end

# for GLMMs we're actually looking at the deviance and additive constants are comparable
# (because GLM deviance is actually part of the GLMM deviance computation)
# for LMMs, we're always looking at the "deviance scale" but without the additive constant
# for the fully saturated model
function _criterion(
    x::Union{GeneralizedLinearModel,TableRegressionModel{<:GeneralizedLinearModel}}
)
    return deviance(x)
end
function _criterion(x::Union{LinearModel,TableRegressionModel{<:LinearModel}})
    return -2 * loglikelihood(x)
end

function likelihoodratiotest(
    m0::Union{
        TableRegressionModel{<:Union{LinearModel,GeneralizedLinearModel}},
        LinearModel,
        GeneralizedLinearModel,
    },
    m::MixedModel...,
)
    _iscomparable(m0, first(m)) ||
        throw(ArgumentError("""Models are not comparable: are the objectives, data
                               and, where appropriate, the link and family the same?
                            """))
    lrt = likelihoodratiotest(m...)
    devs = pushfirst!(lrt.deviance, _criterion(m0))
    formulas = pushfirst!(lrt.formulas, _formula(m0))
    dofs = pushfirst!(lrt.models.dof, dof(m0))
    devdiffs = pushfirst!(lrt.tests.deviancediff, devs[1] - devs[2])
    dofdiffs = pushfirst!(lrt.tests.dofdiff, dofs[2] - dofs[1])

    df, dev = first(dofdiffs), first(devdiffs)
    p = dev > 0 ? ccdf(Chisq(df), dev) : NaN
    pvals = pushfirst!(lrt.tests.pvalues, p)

    return LikelihoodRatioTest(
        formulas,
        (dof=dofs, deviance=devs),
        (dofdiff=dofdiffs, deviancediff=devdiffs, pvalues=pvals),
        lrt.linear,
    )
end

function Base.show(io::IO, ::MIME"text/plain", lrt::LikelihoodRatioTest)
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
    outrows = Matrix{String}(undef, nr + 1, nc)

    outrows[1, :] = [
        "", "model-dof", lrt.linear ? "-2 logLik" : "deviance", "χ²", "χ²-dof", "P(>χ²)"
    ] # colnms

    outrows[2, :] = [
        "[1]", string(lrt.dof[1]), Ryu.writefixed(lrt.deviance[1], 4), " ", " ", " "
    ]

    for i in 2:nr
        outrows[i + 1, :] = [
            "[$i]",
            string(lrt.dof[i]),
            Ryu.writefixed(lrt.deviance[i], 4),
            Ryu.writefixed(Δdev[i - 1], 4),
            string(Δdf[i - 1]),
            string(StatsBase.PValue(lrt.pvalues[i - 1])),
        ]
    end
    colwidths = length.(outrows)
    max_colwidths = [maximum(view(colwidths, :, i)) for i in 1:nc]
    totwidth = sum(max_colwidths) + 2 * 5

    println(io, '─'^totwidth)

    for r in 1:(nr + 1)
        for c in 1:nc
            cur_cell = outrows[r, c]
            cur_cell_len = length(cur_cell)

            padding = " "^(max_colwidths[c] - cur_cell_len)
            if c > 1
                padding = "  " * padding
            end

            print(io, padding)
            print(io, cur_cell)
        end
        print(io, "\n")
        r == 1 && println(io, '─'^totwidth)
    end
    print(io, '─'^totwidth)

    return nothing
end

Base.show(io::IO, lrt::LikelihoodRatioTest) = Base.show(io, MIME"text/plain"(), lrt)

function _iscomparable(m::LinearMixedModel...)
    isconstant(getproperty.(getproperty.(m, :optsum), :REML)) || throw(
        ArgumentError(
            "Models must all be fit with the same objective (i.e. all ML or all REML)"
        ),
    )

    if any(getproperty.(getproperty.(m, :optsum), :REML))
        isconstant(coefnames.(m)) || throw(
            ArgumentError(
                "Likelihood-ratio tests for REML-fitted models are only valid when the fixed-effects specifications are identical"
            ),
        )
    end

    isconstant(nobs.(m)) ||
        throw(ArgumentError("Models must have the same number of observations"))

    return true
end

# XXX we need the where clause to distinguish from the general method
# but static analysis complains if we don't use the type parameter
function _samefamily(
    ::GeneralizedLinearMixedModel{<:AbstractFloat,S}...
) where {S<:Distribution}
    return true
end
_samefamily(::GeneralizedLinearMixedModel...) = false

function _iscomparable(m::GeneralizedLinearMixedModel...)
    # TODO: test that all models are fit with same fast/nAGQ option?
    _samefamily(m...) || throw(ArgumentError("Models must be fit to the same distribution"))

    isconstant(string.(Link.(m))) ||
        throw(ArgumentError("Models must have the same link function"))

    isconstant(nobs.(m)) ||
        throw(ArgumentError("Models must have the same number of observations"))

    return true
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
    all(in.(coefnames(m1), Ref(coefnames(m2)))) || return false

    # check that the same grouping vars occur in the outer model
    grpng1 = fname.(m1.reterms)
    grpng2 = fname.(m2.reterms)

    all(in.(grpng1, Ref(grpng2))) || return false

    # check that every intercept/slope for a grouping var occurs in the
    # same grouping
    re1 = Dict(fname(re) => re.cnames for re in m1.reterms)
    re2 = Dict(fname(re) => re.cnames for re in m2.reterms)

    all(all(in.(val, Ref(re2[key]))) for (key, val) in re1) || return false

    return true
end

function _iscomparable(
    m1::TableRegressionModel{<:Union{LinearModel,GeneralizedLinearModel}}, m2::MixedModel
)
    _iscomparable(m1.model, m2) || return false

    # check that the nested fixef are a subset of the outer
    all(in.(coefnames(m1), Ref(coefnames(m2)))) || return false

    return true
end

# GLM isn't nested with in LMM and LM isn't nested within GLMM
_iscomparable(m1::Union{LinearModel,GeneralizedLinearModel}, m2::MixedModel) = false

function _iscomparable(m1::LinearModel, m2::LinearMixedModel)
    nobs(m1) == nobs(m2) || return false

    # XXX This reaches into the internal structure of GLM
    size(m1.pp.X, 2) <= size(m2.X, 2) || return false

    _isnested(m1.pp.X, m2.X) || return false

    !m2.optsum.REML ||
        throw(ArgumentError("REML-fitted models cannot be compared to linear models"))

    return true
end

function _iscomparable(m1::GeneralizedLinearModel, m2::GeneralizedLinearMixedModel)
    nobs(m1) == nobs(m2) || return false

    size(modelmatrix(m1), 2) <= size(modelmatrix(m2), 2) || return false

    _isnested(modelmatrix(m1), modelmatrix(m2)) || return false

    Distribution(m1) == Distribution(m2) ||
        throw(ArgumentError("Models must be fit to the same distribution"))

    Link(m1) == Link(m2) || throw(ArgumentError("Models must have the same link function"))

    return true
end

"""
    _isnested(x::AbstractMatrix, y::AbstractMatrix; atol::Real=0.0)

Test whether the column span of `x` is a subspace of (nested within)
the column span of y.

The nesting of the column span of the fixed-effects model matrices is a necessary,
but not sufficient condition for a linear model (whether mixed-effects or not)
to be nested within a linear mixed-effects model.

!!! note
    The `rtol` argument is an internal threshold and not currently
    compatible with the `atol` argument of `StatsModels.isnested`.
"""
function _isnested(x::AbstractMatrix, y::AbstractMatrix; rtol=1e-8, ranktol=1e-8)

    # technically this can return false positives if x or y
    # are rank deficient, but either they're rank deficient
    # in the same way (b/c same data) and we don't care OR
    # it's not the same data/fixef specification and we're
    # extra conservative
    size(x, 2) <= size(y, 2) || return false

    qy = qr(y).Q

    qrx = pivoted_qr(x)
    dvec = abs.(diag(qrx.R))
    fdv = first(dvec)
    cmp = fdv * ranktol
    r = searchsortedlast(dvec, cmp; rev=true)

    p = qy' * x

    nested = map(eachcol(p)) do col
        # if set Julia 1.6 as the minimum, we can use last(col, r)
        top = @view col[firstindex(col):(end - r - 1)]
        tail = @view col[(end - r):end]
        return norm(tail) / norm(top) < rtol
    end

    return all(nested)
end
