"""
    LikelihoodRatioTest{N}

Result of `MixedModels.likelihoodratiotest`.

This struct wraps `StatsModels.LRTestResult` with a bit more metadata
to enable a few additional `show` methods.

# Fields
- `formulas::NTuple{N, String}`
- `lrt::StatsModels.LRTestResult{N}`
- `linear::Bool`
"""
Base.@kwdef struct LikelihoodRatioTest{N} <: StatsAPI.HypothesisTest
    formulas::NTuple{N,String}
    lrt::StatsModels.LRTestResult{N}
    linear::Bool
end

"""
    pvalue(lrt::LikelihoodRatioTest)

Extract the p-value associated with a likelihood ratio test.

For `LikelihoodRatioTest`s containing more than one model comparison, i.e. more than two models,
this throws an error because it is unclear which p-value is desired.

To get p-values for multiple tests, use `lrt.pvalues`.
"""
function StatsAPI.pvalue(lrt::LikelihoodRatioTest)
    pvalues = lrt.pvalues
    if length(pvalues) > 2
        throw(
            ArgumentError(
                "Cannot extract **only one** p-value from a multiple test result."
            ),
        )
    end

    return last(pvalues)
end

StatsModels.lrtest(x::LikelihoodRatioTest) = x.lrt

function Base.getproperty(lrt::LikelihoodRatioTest, s::Symbol)
    if s in fieldnames(LikelihoodRatioTest)
        getfield(lrt, s)
    elseif s == :pvalues
        lrt.pval
    else
        getproperty(lrt.lrt, s)
    end
end

const GLM_TYPES = Union{
    TableRegressionModel{<:Union{LinearModel,GeneralizedLinearModel}},
    LinearModel,
    GeneralizedLinearModel,
}

"""
    likelihoodratiotest(m::MixedModel...)
    likelihoodratiotest(m0::LinearModel, m::MixedModel...)
    likelihoodratiotest(m0::GeneralizedLinearModel, m::MixedModel...)
    likelihoodratiotest(m0::TableRegressionModel{LinearModel}, m::MixedModel...)
    likelihoodratiotest(m0::TableRegressionModel{GeneralizedLinearModel}, m::MixedModel...)


Likeihood ratio test applied to a set of nested models.

This function is wrapper around `StatsModels.lrtest` that provides improved `show` methods
that include the model formulae, where available. For mixed models, the [`isnested`](@ref)
functionality may be overly conservative. For example, nested models with "clever" parameterizations
may be incorrectly rejected as being non-negative. In such cases, it is incumbent on the
user to perform the likelihood ratio test manually and construct the resulting [`LikelihoodRatioTest`](@ref)
object.
"""
function likelihoodratiotest(m0::Union{GLM_TYPES,MixedModel},
    m::MixedModel...; kwargs...)
    formulas = (_formula(m0), _formula.(m)...)
    return LikelihoodRatioTest(
        formulas, lrtest(m0, m...; kwargs...), first(m) isa LinearMixedModel
    )
end

_formula(::Union{LinearModel,GeneralizedLinearModel}) = "NA"
function _formula(x::TableRegressionModel{<:Union{LinearModel,GeneralizedLinearModel}})
    return String(Symbol(x.mf.f))
end
_formula(x::MixedModel) = string(formula(x))

Base.show(io::IO, lrt::LikelihoodRatioTest) = show(io, MIME("text/plain"), lrt)
function Base.show(io::IO, ::MIME"text/plain", lrt::LikelihoodRatioTest{N}) where {N}
    println(io, "Likelihood-ratio test: $N models fitted on $(lrt.nobs) observations")
    println(io, "Model Formulae")
    for (i, f) in enumerate(lrt.formulas)
        println(io, "$i: $f")
    end

    # adapted from https://github.com/JuliaStats/StatsModels.jl/blob/ad89c6755066511e99148f013e81437e29459ed2/src/lrtest.jl#L130-L178

    Δdf = _diff(lrt.dof)
    chisq = abs.(2 .* _diff(lrt.loglikelihood))
    linear = lrt.linear

    nc = 6
    nr = N
    outrows = Matrix{String}(undef, nr + 1, nc)

    outrows[1, :] = ["",
        "DoF",
        "-2 logLik",
        "χ²",
        "χ²-dof", "P(>χ²)"]
    outrows[2, :] = ["[1]",
        @sprintf("%.0d", lrt.dof[1]),
        @sprintf("%.4f", -2 * lrt.loglikelihood[1]),
        "",
        "",
        ""]

    for i in 2:nr
        outrows[i + 1, :] = ["[$i]",
            @sprintf("%.0d", lrt.dof[i]),
            @sprintf("%.4f", -2 * lrt.loglikelihood[i]),
            @sprintf("%.4f", chisq[i - 1]),
            @sprintf("%.0d", Δdf[i - 1]),
            string(StatsBase.PValue(lrt.pval[i]))]
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

#####
##### StatsModels methods for lrtest and isnested
#####

_diff(t::NTuple{N}) where {N} = ntuple(i -> t[i + 1] - t[i], N - 1)

# adapted from StatsModels with the type check elided
# https://github.com/JuliaStats/StatsModels.jl/blob/ad89c6755066511e99148f013e81437e29459ed2/src/lrtest.jl#L74-L128
# we only accept one non mixed GLM so that we can require a MixedModel and avoid piracy
# we always do forward selection for GLM/GLMM comparisons
# because we need the GLM first for dispatch reasons
function StatsModels.lrtest(m0::GLM_TYPES, m::MixedModel...; atol::Real=0.0)
    mods = (m0, m...)
    length(mods) >= 2 ||
        throw(ArgumentError("At least two models are needed to perform LR test"))
    df = dof.(mods)
    all(==(nobs(mods[1])), nobs.(mods)) ||
        throw(
            ArgumentError(
                "LR test is only valid for models fitted on the same data, " *
                "but number of observations differ",
            ),
        )
    for i in 2:length(mods)
        if df[i - 1] >= df[i] || !isnested(mods[i - 1], mods[i]; atol=atol)
            throw(ArgumentError("LR test is only valid for nested models"))
        end
    end

    dev = deviance.(mods)

    Δdf = (NaN, _diff(df)...)
    dfr = Int.(dof_residual.(mods))

    ll = loglikelihood.(mods)
    chisq = (NaN, 2 .* abs.(_diff(ll))...)

    for i in 2:length(ll)
        ll[i - 1] > ll[i] && !isapprox(ll[i - 1], ll[i]; atol=atol) &&
            throw(
                ArgumentError(
                    "Log-likelihood must not be lower " *
                    "in models with more degrees of freedom",
                ),
            )
    end

    pval = chisqccdf.(abs.(Δdf), chisq)
    return StatsModels.LRTestResult(Int(nobs(mods[1])), dev, ll, df, pval)
end

"""
    isnested(m1::MixedModel, m2::MixedModel; atol::Real=0.0)

Indicate whether model `m1` is nested in model `m2`, i.e. whether `m1` can be obtained by constraining some parameters in `m2`.
Both models must have been fitted on the same data.
This check is conservative for `MixedModel`s and may reject nested models with different parameterizations as being non nested.
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

function StatsModels.isnested(
    m1::TableRegressionModel{<:Union{<:GeneralizedLinearModel,<:LinearModel}},
    m2::MixedModel;
    atol::Real=0.0,
)
    return _iscomparable(m1, m2) && isnested(m1.model, m2; atol)
end

function StatsModels.isnested(m1::LinearModel, m2::LinearMixedModel; atol::Real=0.0)
    nobs(m1) == nobs(m2) || return false

    size(modelmatrix(m1), 2) <= size(modelmatrix(m2), 2) || return false

    _isnested(modelmatrix(m1), modelmatrix(m2)) || return false

    !m2.optsum.REML ||
        throw(
            ArgumentError(
                "REML-fitted modMixedModels.isnested(lm0, fm1)els cannot be compared to linear models"
            ),
        )

    return true
end

function StatsModels.isnested(
    m1::GeneralizedLinearModel, m2::GeneralizedLinearMixedModel; atol::Real=0.0
)
    nobs(m1) == nobs(m2) || return false

    size(modelmatrix(m1), 2) <= size(modelmatrix(m2), 2) || return false

    _isnested(modelmatrix(m1), modelmatrix(m2)) || return false

    Distribution(m1) == Distribution(m2) ||
        throw(ArgumentError("Models must be fit to the same distribution"))

    Link(m1) == Link(m2) || throw(ArgumentError("Models must have the same link function"))

    return true
end

#####
##### Helper functions for isnested
#####

"""
    _iscomparable(m::LinearMixedModel...)


Check whether LMMs are comparable on the basis of their REML criterion.
"""
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

    return true
end

"""
    _iscomparable(m::GeneralizedLinearMixedModel...)


Check whether GLMMs are comparable in terms of their model families and links.
"""
function _iscomparable(m::GeneralizedLinearMixedModel...)
    # TODO: test that all models are fit with same fast/nAGQ option?
    _samefamily(m...) || throw(ArgumentError("Models must be fit to the same distribution"))

    isconstant(string.(Link.(m))) ||
        throw(ArgumentError("Models must have the same link function"))

    return true
end

"""
    _iscomparable(m1::TableRegressionModel, m2::MixedModel)_samefamily(
    ::GeneralizedLinearMixedModel

Check whether a TableRegressionModel and a MixedModel have coefficient names indicative of nesting.
"""
function _iscomparable(
    m1::TableRegressionModel{<:Union{LinearModel,GeneralizedLinearModel}}, m2::MixedModel
)
    # check that the nested fixef are a subset of the outer
    all(in.(coefnames(m1), Ref(coefnames(m2)))) || return false

    return true
end

"""
    _samefamily(::GeneralizedLinearMixedModel...)

Check whether all GLMMS come from the same model family.
"""
function _samefamily(
    ::GeneralizedLinearMixedModel{<:AbstractFloat,S}...
) where {S<:Distribution}
    # XXX we need the where clause to distinguish from the general method
    # but static analysis complains if we don't use the type parameter
    return true
end
_samefamily(::GeneralizedLinearMixedModel...) = false

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
