"""
    GeneralizedLinearMixedModel

Generalized linear mixed-effects model representation

# Fields
- `LMM`: a [`LinearMixedModel`](@ref) - the local approximation to the GLMM.
- `β`: the pivoted and possibly truncated fixed-effects vector
- `β₀`: similar to `β`. Used in the PIRLS algorithm if step-halving is needed.
- `θ`: covariance parameter vector
- `ϕ`: dispersion parameter, as a zero- or one-element vector. It is empty when ϕ is
  plugged in from the Pearson moment estimator (`fast=true`, and always for families
  without a dispersion parameter) and holds a single value when ϕ is a free parameter
  of the outer optimization (`fast=false` with a dispersion family). See
  [`_dispersion`](@ref).
- `b`: similar to `u`, equivalent to `broadcast!(*, b, LMM.Λ, u)`
- `u`: a vector of matrices of random effects
- `u₀`: similar to `u`.  Used in the PIRLS algorithm if step-halving is needed.
- `resp`: a `GlmResp` object
- `η`: the linear predictor
- `wt`: vector of prior case weights, a value of `T[]` indicates equal weights.
The following fields are used in adaptive Gauss-Hermite quadrature, which applies
only to models with a single random-effects term, in which case their lengths are
the number of levels in the grouping factor for that term.  Otherwise they are
zero-length vectors.
- `devc`: vector of deviance components
- `devc0`: vector of deviance components at offset of zero
- `sd`: approximate standard deviation of the conditional density
- `mult`: multiplier

# Properties

In addition to the fieldnames, the following names are also accessible through the `.` extractor

- `theta`: synonym for `θ`
- `beta`: synonym for `β`
- `σ` or `sigma`: common scale parameter (value is `NaN` for distributions without a scale parameter)
- `formula`, `trms`, `A`, `L`, and `optsum`: fields of the `LMM` field
- `X`: fixed-effects model matrix
- `y`: response vector

"""
struct GeneralizedLinearMixedModel{T<:AbstractFloat,D<:Distribution} <: MixedModel{T}
    LMM::LinearMixedModel{T}
    β::Vector{T}
    β₀::Vector{T}
    θ::Vector{T}
    ϕ::Vector{T}
    b::Vector{Matrix{T}}
    u::Vector{Matrix{T}}
    u₀::Vector{Matrix{T}}
    resp::GLM.GlmResp
    η::Vector{T}
    wt::Vector{T}
    devc::Vector{T}
    devc0::Vector{T}
    sd::Vector{T}
    mult::Vector{T}
end

function StatsAPI.coef(m::GeneralizedLinearMixedModel{T}) where {T}
    piv = pivot(m)
    return invpermute!(copyto!(fill(T(-0.0), length(piv)), m.β), piv)
end

function StatsAPI.coeftable(m::GeneralizedLinearMixedModel)
    co = coef(m)
    se = stderror(m)
    z = co ./ se
    pvalue = ccdf.(Chisq(1), abs2.(z))
    return CoefTable(
        hcat(co, se, z, pvalue),
        ["Coef.", "Std. Error", "z", "Pr(>|z|)"],
        coefnames(m),
        4, # pvalcol
        3, # teststatcol
    )
end

"""
    deviance(m::GeneralizedLinearMixedModel{T}, nAGQ=1)::T where {T}

Return the deviance of `m` evaluated by the Laplace approximation (`nAGQ=1`)
or `nAGQ`-point adaptive Gauss-Hermite quadrature.

If the distribution `D` does not have a scale parameter the Laplace approximation
is the squared length of the conditional modes, ``u``, plus the determinant
of ``Λ'Z'WZΛ + I``, plus the sum of the squared deviance residuals.

For distributions with a dispersion parameter ``ϕ``, the value of ϕ comes from
`_dispersion` and is shared with `dispersion` and `loglikelihood`, so the three
always agree. Depending on how the model was fit, that is either the moment
estimator ``pwrss(m) / nobs(m)`` (`fast=true`) or a free parameter of the outer
optimization (`fast=false`). See the internal `_laplace_deviance` and
`_agq_deviance` for the exact expressions.
"""
function StatsAPI.deviance(m::GeneralizedLinearMixedModel{T}, nAGQ=1) where {T}
    return nAGQ == 1 ? _laplace_deviance(m) : _agq_deviance(m, nAGQ)
end

StatsAPI.deviance(m::GeneralizedLinearMixedModel) = deviance(m, m.optsum.nAGQ)

"""
    _laplace_deviance(m::GeneralizedLinearMixedModel)

Internal Laplace-approximation deviance (equivalent to AGQ with `nAGQ == 1`)
used as the NLopt/PRIMA objective when fitting with the Laplace approximation.

For non-dispersion families (Bernoulli, Binomial, Poisson) this is
`sum(devresid) + sum(u²) + logdet(m)`. For dispersion families it is

    -2 · Σ loglik_obs(d, yᵢ, μᵢ, wᵢ, ϕ) + sum(u²)/ϕ + logdet(m)

Note the `1/ϕ` on the penalty: `u ~ N(0, ϕI)` under this package's relative
covariance parameterization (see [`deviance!`](@ref)), so `sum(u²)` is on the ϕ
scale too. Specializing the expression to `Normal` with an identity link recovers
`objective(::LinearMixedModel)` exactly, which is the check that fixes the
normalization. The two branches are byte-equivalent for non-dispersion families,
where ϕ ≡ 1.
"""
function _laplace_deviance(m::GeneralizedLinearMixedModel{T}) where {T}
    uss = sum(u -> sum(abs2, u), m.u)
    ld = logdet(m)
    if dispersion_parameter(m.resp.d)
        ϕ = _dispersion(m)
        return T(-2 * _loglik_data(m.resp, ϕ) + uss / ϕ + ld)
    end
    return T(sum(m.resp.devresid) + ld + uss)
end

"""
    _dispersion(m::GeneralizedLinearMixedModel{T})

The value of ϕ used by the objective, the log-likelihood and [`dispersion`](@ref).

Families without a dispersion parameter give `one(T)`. Otherwise there are two
regimes, distinguished by whether `m.ϕ` is empty:

  - `isempty(m.ϕ)` — ϕ is plugged in from the Pearson moment estimator
    `pwrss(m) / nobs(m)`. This is the `fast=true` regime, and it is what every
    dispersion fit did before ϕ became an outer parameter.
  - otherwise — ϕ is a free parameter of the outer optimization and `first(m.ϕ)`
    is its current value. This is the `fast=false` regime.

The two coincide only for `Normal`: there `pwrss/n` really is the conditional MLE
of ϕ, so the outer optimization just rediscovers it. For `Gamma` and
`InverseGaussian` the conditional MLE solves a digamma equation rather than a
moment condition, and the free parameter converges to that instead — the moment
estimator is then a genuinely different (and lme4-compatible) answer.

Either way ϕ does not affect the conditional modes; see [`deviance!`](@ref).
"""
function _dispersion(m::GeneralizedLinearMixedModel{T}) where {T}
    dispersion_parameter(m.resp.d) || return one(T)
    isempty(m.ϕ) && return max(pwrss(m) / nobs(m), eps(T))
    return max(first(m.ϕ), eps(T))
end

"""
    _agq_deviance(m::GeneralizedLinearMixedModel, nAGQ)

Internal adaptive Gauss-Hermite quadrature deviance, used as the NLopt/PRIMA
objective for `nAGQ > 1`.

For dispersion families ϕ comes from [`_dispersion`](@ref) and enters in three
places, all of which reduce to no-ops when ϕ ≡ 1:

  - the per-group integrand is `D_g(u) = (u² + Σ_{i∈g} devresid_i(u))/ϕ`, the
    whole penalized quantity on the ϕ scale rather than just the data part;
  - the quadrature nodes are spaced by `√ϕ / L.diag`, since the conditional
    standard deviation of `u` given `y` carries the same ϕ as the prior;
  - the μ-independent normaliser is
    `Cϕ = -2·Σ loglik_obs(d, yᵢ, μᵢ, wᵢ, ϕ) - sum(devresid)/ϕ + nᵤ·log ϕ`,
    computed once at the mode. The trailing `nᵤ·log ϕ` compensates exactly for
    the `√ϕ` in the node spacing, which would otherwise contribute `-nᵤ·log ϕ`
    through the `sum(log, sd)` Jacobian term.

The AGQ deviance is then `sum(D_g(û)) - 2·(sum(log,mult) + sum(log,sd)) + Cϕ`.
For non-dispersion families ϕ ≡ 1 and Cϕ ≡ 0, reducing exactly to the prior
formula, bit for bit.

By construction `_agq_deviance(m, 1) == _laplace_deviance(m)` (the Laplace
approximation *is* AGQ with n=1); that identity is what pins down the constants
above and is checked in the test suite.
"""
function _agq_deviance(m::GeneralizedLinearMixedModel{T}, nAGQ) where {T}
    u = vec(first(m.u))
    u₀ = vec(first(m.u₀))
    copyto!(u₀, u)
    ra = RaggedArray(m.resp.devresid, first(m.LMM.reterms).refs)

    has_disp = dispersion_parameter(m.resp.d)
    ϕ = _dispersion(m)
    # Cϕ is the μ-independent part of -2·Σ loglik_obs(d, y, μ, w, ϕ); the
    # identity -2·loglik_obs = devresid/ϕ + c(y, w, ϕ) lets us evaluate it
    # once at the mode and treat it as constant across quadrature nodes. For
    # non-dispersion families we keep Cϕ = 0 so existing nAGQ>1 fits stay
    # bit-identical (Binomial/Poisson would otherwise pick up a constant
    # saturated-likelihood term that the historical formula dropped).
    Cϕ = if has_disp
        T(-2 * _loglik_data(m.resp, ϕ) - sum(m.resp.devresid) / ϕ +
          length(u) * log(ϕ))
    else
        zero(T)
    end

    # devc0_g = (u_g² + Σ_{i∈g} devresid_i)/ϕ  at u = û
    sum!(fill!(m.devc0, 0), ra)
    @. m.devc0 = (abs2(u) + m.devc0) / ϕ
    devc0 = m.devc0

    # the conditional sd of u given y is √ϕ/L.diag; the √ϕ is undone by the
    # nᵤ·log ϕ carried in Cϕ above
    sd = map!(inv, m.sd, first(m.LMM.L).diag)
    has_disp && (sd .*= sqrt(ϕ))
    mult = fill!(m.mult, 0)
    devc = m.devc
    for (z, w) in GHnorm(nAGQ)
        if !iszero(w)
            if iszero(z)
                mult .+= w
            else
                @. u = u₀ + z * sd
                updateη!(m)
                sum!(fill!(devc, 0), ra)
                @. devc = (abs2(u) + devc) / ϕ
                @. mult += exp((abs2(z) + devc0 - devc) / 2) * w
            end
        end
    end
    copyto!(u, u₀)
    updateη!(m)
    return sum(devc0) - 2 * (sum(log, mult) + sum(log, sd)) + Cϕ
end

fixef(m::GeneralizedLinearMixedModel) = m.β

function fixef!(v::AbstractVector{Tv}, m::GeneralizedLinearMixedModel{T}) where {Tv,T}
    return copyto!(fill!(v, -zero(Tv)), m.β)
end

objective(m::GeneralizedLinearMixedModel) = deviance(m)

"""
    GLM.wrkresp!(v::AbstractVector{T}, resp::GLM.GlmResp{AbstractVector{T}})

A copy of a method from GLM that generalizes the types in the signature
"""
function GLM.wrkresp!(
    v::AbstractVector{T}, r::GLM.GlmResp{Vector{T}}
) where {T<:AbstractFloat}
    v .= r.eta .+ r.wrkresid
    isempty(r.offset) && return v
    return v .-= r.offset
end

"""
    deviance!(m::GeneralizedLinearMixedModel, nAGQ=1)

Update `m.η`, `m.μ`, etc., install the working response and working weights in
`m.LMM`, update `m.LMM.A` and `m.LMM.R`, then evaluate the `deviance`.

Note that PIRLS itself does not depend on ϕ. Under the covariance parameterization
this package uses, `θ` is *relative* to the scale parameter — `Var(b) = ϕΛΛ'`, i.e.
`u ~ N(0, ϕI)`, matching [`LinearMixedModel`](@ref) and the `σ` factor in
[`σs`](@ref) — so the penalized objective is `(Σᵢ wᵢrᵢ² + ‖u‖²)/ϕ` and the `1/ϕ`
factors straight out of the minimization over `u`. ϕ enters the deviance, not the
conditional modes.
"""
function deviance!(m::GeneralizedLinearMixedModel, nAGQ=1)
    updateη!(m)
    GLM.wrkresp!(m.LMM.y, m.resp)
    reweight!(m.LMM, m.resp.wrkwt)
    return deviance(m, nAGQ)
end

function GLM.dispersion(m::GeneralizedLinearMixedModel{T}, sqr::Bool=false) where {T}
    dispersion_parameter(m.resp.d) || return one(T)
    # Either the Pearson moment estimator pwrss/n (which matches lme4's `sigma()`
    # and the LMM convention at `varest(::LinearMixedModel)`) or the free outer
    # parameter, depending on how the model was fit.  See `_dispersion`.
    s2 = _dispersion(m)
    return sqr ? s2 : sqrt(s2)
end

GLM.dispersion_parameter(m::GeneralizedLinearMixedModel) = dispersion_parameter(m.resp.d)

Distributions.Distribution(m::GeneralizedLinearMixedModel{T,D}) where {T,D} = D

function StatsAPI.fit(
    ::Type{GeneralizedLinearMixedModel},
    f::FormulaTerm,
    tbl,
    d::Distribution=Normal(),
    l::Link=canonicallink(d);
    kwargs...,
)
    return fit(GeneralizedLinearMixedModel, f, columntable(tbl), d, l; kwargs...)
end

function StatsAPI.fit(
    ::Type{GeneralizedLinearMixedModel},
    f::FormulaTerm,
    tbl::Tables.ColumnTable,
    d::Distribution,
    l::Link=canonicallink(d);
    weights=[],
    wts=nothing,
    contrasts=Dict{Symbol,Any}(),
    offset=[],
    amalgamate=true,
    kwargs...,
)
    return fit!(
        GeneralizedLinearMixedModel(
            f, tbl, d, l; weights, wts, offset, contrasts, amalgamate
        );
        kwargs...,
    )
end

"""
    glmm(args...; kwargs...)

Convenience wrapper for `fit(GeneralizedLinearMixedModel, args...; kwargs...)`.

See [`GeneralizedLinearMixedModel`](@ref) and [`fit!`](@ref) for more information.
"""
glmm(args...; kwargs...) = fit(GeneralizedLinearMixedModel, args...; kwargs...)

function StatsAPI.fit(
    ::Type{MixedModel},
    f::FormulaTerm,
    tbl,
    d::Distribution,
    l::Link=canonicallink(d);
    kwargs...,
)
    return fit(GeneralizedLinearMixedModel, f, tbl, d, l; kwargs...)
end

"""
    fit!(m::GeneralizedLinearMixedModel; fast=false, nAGQ=1,
                                         verbose=false, progress=true,
                                         init_from_lmm=Set())

Optimize the objective function for `m`.

When `fast` is `true` a potentially much faster but slightly less accurate algorithm, in
which `pirls!` optimizes both the random effects and the fixed-effects parameters,
is used.

If `progress` is `true`, the default, a `ProgressMeter.ProgressUnknown` counter is displayed.
during the iterations to minimize the deviance.  There is a delay before this display is initialized
and it may not be shown at all for models that are optimized quickly.

If `verbose` is `true`, then both the intermediate results of both the nonlinear optimization and PIRLS are also displayed on standard output.

By default, the starting values for model fitting are taken from a (non mixed,
i.e. marginal ) GLM fit. Experience with larger datasets (many thousands of
observations and/or hundreds of levels of the grouping variables) has suggested
that fitting a (Gaussian) linear mixed model on the untransformed data may
provide better starting values and thus overall faster fits even though an
entire LMM must be fit before the GLMM can be fit. `init_from_lmm` can be used
to specify which starting values from an LMM to use. Valid options are any
collection (array, set, etc.) containing one or more of `:β` and `:θ`, the
default is the empty set.

!!! note
    Initializing from an LMM requires fitting the entire LMM first, so when
    `progress=true`, there will be two progress bars: first for the LMM, then
    for the GLMM.

!!! warning
    The `init_from_lmm` functionality is experimental and may change or be removed entirely
    without being considered a breaking change.
"""
function StatsAPI.fit!(
    m::GeneralizedLinearMixedModel{T};
    verbose::Bool=false,
    fast::Bool=false,
    nAGQ::Integer=1,
    progress::Bool=true,
    init_from_lmm=Set(),
    backend::Symbol=m.optsum.backend,
    optimizer::Symbol=m.optsum.optimizer,
) where {T}
    β = copy(m.β)
    θ = copy(m.θ)
    lm = m.LMM
    optsum = lm.optsum

    issubset(init_from_lmm, [:θ, :β]) ||
        throw(ArgumentError("Invalid parameter selection for init_from_lmm"))

    if optsum.feval > 0
        throw(ArgumentError("This model has already been fitted. Use refit!() instead."))
    end

    disp = dispersion_parameter(m.resp.d)
    if disp
        @info "Fitting a GLMM with a dispersion parameter. " *
            (
                if fast
                    "ϕ is plugged in from the Pearson moment estimator pwrss/n, which " *
                    "matches lme4's `sigma()`. "
                else
                    "ϕ is estimated jointly with β and θ as a parameter of the outer " *
                    "optimization. "
                end
            ) *
            "Please report any discrepancies vs lme4."
    end

    if all(==(first(m.y)), m.y)
        throw(ArgumentError("The response is constant and thus model fitting has failed"))
    end

    if !isempty(init_from_lmm)
        fit!(lm; progress)
        :θ in init_from_lmm && copyto!(θ, lm.θ)
        :β in init_from_lmm && copyto!(β, lm.β)
        unfit!(lm)
    end

    # ϕ is a free parameter of the outer optimization only for `fast=false` fits
    # of a dispersion family; otherwise it is plugged in from pwrss/n.  Emptying
    # it here rather than relying on the constructor keeps `refit!` honest when
    # the same model is refit with a different `fast`.
    empty!(m.ϕ)
    if !fast
        optsum.initial = vcat(β, lm.optsum.final)
        if disp
            # a plug-in evaluation at the starting β/θ gives a much better
            # starting ϕ than any fixed constant would
            pirls!(setβθ!(m, optsum.initial), false, verbose)
            push!(m.ϕ, max(pwrss(m) / nobs(m), eps(T)))
            optsum.initial = vcat(optsum.initial, log(first(m.ϕ)))
        end
        optsum.final = copy(optsum.initial)
    end

    optsum.backend = backend
    optsum.optimizer = optimizer

    xmin, fmin = optimize!(m; progress, fast, verbose, nAGQ)

    θopt = if length(xmin) == length(θ)
        xmin
    else
        view(xmin, (length(β) + 1):(length(β) + length(θ)))
    end
    rectify!(m.LMM)                  # flip signs of columns of m.λ elements with negative diagonal els
    getθ!(θopt, m)                   # use the rectified values in xmin

    ## check if very small parameter values bounded below by zero can be set to zero
    xmin_ = copy(xmin)
    # log ϕ is unbounded, so a lower bound of -Inf keeps it out of the
    # zero-snapping loop below -- snapping it would send ϕ to 1, not to 0
    lb = fast ? lowerbd(m) : vcat(zero(β), lowerbd(m))
    isempty(m.ϕ) || (lb = vcat(lb, T(-Inf)))
    for i in eachindex(xmin_)
        if iszero(lb[i]) && zero(T) < xmin_[i] < optsum.xtol_zero_abs
            xmin_[i] = zero(T)
        end
    end
    if xmin ≠ xmin_
        if (zeroobj = objective!(m, xmin_; nAGQ, fast, verbose)) ≤
            (fmin + optsum.ftol_zero_abs)
            fmin = zeroobj
            copyto!(xmin, xmin_)
            push!(optsum.fitlog, (; θ=copy(xmin), objective=fmin))
        end
    end

    ## ensure that the parameter values saved in m are xmin
    objective!(m, xmin; fast, verbose, nAGQ)
    optsum.final = xmin
    optsum.fmin = fmin
    optsum.nAGQ = nAGQ
    return m
end

StatsAPI.fitted(m::GeneralizedLinearMixedModel) = m.resp.mu

function GeneralizedLinearMixedModel(
    f::FormulaTerm,
    tbl,
    d::Type,
    args...;
    kwargs...,
)
    return throw(
        ArgumentError("Expected a Distribution instance (`$d()`), got a type (`$d`).")
    )
end

function GeneralizedLinearMixedModel(
    f::FormulaTerm,
    tbl,
    d::Distribution,
    l::Type;
    kwargs...,
)
    return throw(ArgumentError("Expected a Link instance (`$l()`), got a type (`$l`)."))
end

function GeneralizedLinearMixedModel(
    f::FormulaTerm,
    tbl,
    d::Distribution,
    l::Link=canonicallink(d);
    kwargs...,
)
    return GeneralizedLinearMixedModel(
        f, Tables.columntable(tbl), d, l; kwargs...
    )
end

function GeneralizedLinearMixedModel(
    f::FormulaTerm,
    tbl::Tables.ColumnTable,
    d::Normal,
    l::IdentityLink;
    kwargs...,
)
    return throw(
        ArgumentError("use LinearMixedModel for Normal distribution with IdentityLink")
    )
end

function GeneralizedLinearMixedModel(
    f::FormulaTerm,
    tbl::Tables.ColumnTable,
    d::Distribution,
    l::Link=canonicallink(d);
    weights=[],
    wts=nothing,
    offset=[],
    contrasts=Dict{Symbol,Any}(),
    amalgamate=true,
)
    if wts !== nothing
        Base.depwarn(
            "`wts` keyword argument is deprecated, use `weights` instead",
            :GeneralizedLinearMixedModel,
        )
        weights = wts
    end

    if isa(d, Binomial) && isempty(weights)
        d = Bernoulli()
    end
    (isa(d, Normal) && isa(l, IdentityLink)) && throw(
        ArgumentError("use LinearMixedModel for Normal distribution with IdentityLink")
    )

    LMM = LinearMixedModel(f, tbl; contrasts, weights, amalgamate)
    y = copy(LMM.y)
    constresponse = all(==(first(y)), y)
    # the sqrtwts field must be the correct length and type but we don't know those
    # until after the model is constructed if wt is empty.  Because a LinearMixedModel
    # type is immutable, another one must be created.
    if isempty(weights)
        LMM = LinearMixedModel(
            LMM.formula,
            LMM.reterms,
            LMM.Xymat,
            LMM.feterm,
            fill!(similar(y), 1),
            LMM.parmap,
            LMM.dims,
            LMM.A,
            LMM.L,
            LMM.optsum,
        )
    end
    X = fullrankx(LMM.feterm)
    # if the response is constant, there's no point (and this may even fail)
    # we allow this instead of simply failing so that a constant response can
    # be used as the starting point to simulation where the response will be
    # overwritten before fitting
    constresponse || updateL!(LMM)
    # fit a glm to the fixed-effects only
    T = eltype(LMM.Xymat)
    # newer versions of GLM (>1.8.0) have a kwarg dropcollinear=true
    # which creates problems for the empty fixed-effects case during fitting
    # so just don't allow fitting
    # XXX unfortunately, this means we have double-rank deficiency detection
    # TODO: construct GLM by hand so that we skip collinearity checks
    # TODO: extend this so that we never fit a GLM when initializing from LMM
    dofit = size(X, 2) != 0 # GLM.jl kwarg
    wtkwarg = pkgversion(GLM) >= v"1.9.1" ? :weights : :wts
    weights = convert(Vector{T}, weights)
    gl = glm(X, y, d, l;
        wtkwarg => pkgversion(GLM) >= v"1.9.1" ? FrequencyWeights(weights) : weights,
        dofit,
        :offset => convert(Vector{T}, offset))
    β = dofit ? coef(gl) : T[]
    u = [fill(zero(eltype(y)), vsize(t), nlevs(t)) for t in LMM.reterms]
    # vv is a template vector used to initialize fields for AGQ
    # it is empty unless there is a single random-effects term
    vv = length(u) == 1 ? vec(first(u)) : similar(y, 0)

    res = GeneralizedLinearMixedModel{T,typeof(d)}(
        LMM,
        β,
        copy(β),
        LMM.θ,
        T[],   # ϕ: empty until `fit!` promotes it to a free parameter
        copy.(u),
        u,
        zero.(u),
        gl.rr,
        similar(y),
        weights,
        similar(vv),
        similar(vv),
        similar(vv),
        similar(vv),
    )

    # if the response is constant, there's no point (and this may even fail)
    constresponse || try
        deviance!(res, 1)
    catch ex
        ex isa PosDefException || rethrow()
        @warn "Evaluation at default initial parameter vector failed, " *
            "initializing to very small variances. This may result in long " *
            "model fitting times. You will probably also need to use " *
            "`init_from_lmm=[:β, :θ]` in order to fit the model."
        res.optsum.initial[res.optsum.initial .!= 0] .= 1e-8
    end

    return res
end

function Base.getproperty(m::GeneralizedLinearMixedModel, s::Symbol)
    if s == :theta
        m.θ
    elseif s == :coef
        coef(m)
    elseif s == :beta
        m.β
    elseif s == :objective
        objective(m)
    elseif s ∈ (:σ, :sigma)
        sdest(m)
    elseif s == :σs
        σs(m)
    elseif s == :σρs
        σρs(m)
    elseif s == :y
        m.resp.y
    elseif !hasfield(GeneralizedLinearMixedModel, s) && s ∈ propertynames(m.LMM, true)
        # automatically delegate as much as possible to the internal local linear approximation
        # NB: the !hasfield call has to be first since we're calling getproperty() with m.LMM...
        getproperty(m.LMM, s)
    else
        getfield(m, s)
    end
end

# this copy behavior matches the implicit copy behavior
# for LinearMixedModel. So this is then different than m.θ,
# which returns a reference to the same array
getθ(m::GeneralizedLinearMixedModel) = copy(m.θ)
getθ!(v::AbstractVector{T}, m::GeneralizedLinearMixedModel{T}) where {T} = copyto!(v, m.θ)

StatsAPI.islinear(m::GeneralizedLinearMixedModel) = isa(GLM.Link, GLM.IdentityLink)

GLM.Link(m::GeneralizedLinearMixedModel) = GLM.Link(m.resp)

function StatsAPI.loglikelihood(m::GeneralizedLinearMixedModel{T}) where {T}
    # `_dispersion` is the single source of ϕ̂, so the loglikelihood, deviance and
    # dispersion estimate are always evaluated at the same value whichever regime
    # the model was fit in. For families without a dispersion parameter it returns
    # 1 and `loglik_obs` ignores the ϕ argument anyway.
    r = m.resp
    ϕ = _dispersion(m)
    ll = _loglik_data(r, ϕ)
    uss = sum(u -> sum(abs2, u), m.u)
    return ll - (uss / ϕ + logdet(m)) / 2
end

function _loglik_data(r::GLM.GlmResp, ϕ)
    # Sum of GLM.loglik_obs(d, yᵢ, μᵢ, wᵢ, ϕ); covers Gaussian/Gamma/IG with a
    # profiled ϕ as well as Bernoulli/Binomial/Poisson where ϕ is ignored.
    accum = zero(eltype(r.mu))
    y = r.y
    mu = r.mu
    wts = r.wts
    d = r.d
    if length(wts) == length(y)
        @inbounds for i in eachindex(y, mu, wts)
            accum += GLM.loglik_obs(d, y[i], mu[i], wts[i], ϕ)
        end
    else
        @inbounds for i in eachindex(y, mu)
            accum += GLM.loglik_obs(d, y[i], mu[i], 1, ϕ)
        end
    end
    return accum
end

"""
    lowerbd(m::GeneralizedLinearMixedModel)

Return the vector of _canonical_ lower bounds on the parameters, `θ`.

Note that this method does not distinguish between constrained optimization and
unconstrained optimization with post-fit canonicalization.
"""
lowerbd(m::GeneralizedLinearMixedModel) = lowerbd(m.LMM)

"""
    pwrss(m::GeneralizedLinearMixedModel)

The penalized, weighted residual sum-of-squares for the working LMM at the
current PIRLS state. Equal to `Σ wrkwtᵢ · wrkresidᵢ² + Σⱼ ‖uⱼ‖²` after PIRLS
converges, so `pwrss(m) - sum(abs2, u)` is the Pearson chi-square contribution
used to estimate the dispersion parameter.

This holds regardless of how ϕ is being estimated, since PIRLS does not depend on
ϕ (see [`deviance!`](@ref)). For a model fit with `fast=false` and a dispersion
family, comparing `pwrss(m) / nobs(m)` against [`dispersion`](@ref) contrasts the
moment estimator with the conditional MLE that the outer optimization finds.
"""
pwrss(m::GeneralizedLinearMixedModel) = pwrss(m.LMM)

# Base.Fix1 doesn't forward kwargs
function objective!(m::GeneralizedLinearMixedModel; fast=false, kwargs...)
    return x -> _objective!(m, x, Val(fast); kwargs...)
end

function objective!(m::GeneralizedLinearMixedModel{T}, x; fast=false, kwargs...) where {T}
    return _objective!(m, x, Val(fast); kwargs...)
end

# normally, it doesn't make sense to move a simple branch to dispatch
# HOWEVER, this winds up getting called in optimization a lot and
# moving this to a realization here allows us to avoid dynamic dispatch on setθ! / setθβ!
function _objective!(
    m::GeneralizedLinearMixedModel{T}, x, ::Val{true}; nAGQ=1, verbose=false
) where {T}
    pirls!(setθ!(m, x), true, verbose)
    return nAGQ == 1 ? _laplace_deviance(m) : _agq_deviance(m, nAGQ)
end

function _objective!(
    m::GeneralizedLinearMixedModel{T}, x, ::Val{false}; nAGQ=1, verbose=false
) where {T}
    pirls!(setβθ!(m, x), false, verbose)
    return nAGQ == 1 ? _laplace_deviance(m) : _agq_deviance(m, nAGQ)
end

function Base.propertynames(m::GeneralizedLinearMixedModel, private::Bool=false)
    return (
        :A,
        :L,
        :theta,
        :beta,
        :coef,
        :λ,
        :σ,
        :sigma,
        :X,
        :y,
        :objective,
        :σρs,
        :σs,
        :corr,
        :vcov,
        :PCA,
        :rePCA,
        (
            if private
                fieldnames(GeneralizedLinearMixedModel)
            else
                (:LMM, :β, :θ, :b, :u, :resp, :wt)
            end
        )...,
    )
end

"""
    pirls!(m::GeneralizedLinearMixedModel)

Use Penalized Iteratively Reweighted Least Squares (PIRLS) to determine the conditional
modes of the random effects.

When `varyβ` is true both `u` and `β` are optimized with PIRLS.  Otherwise only `u` is
optimized and `β` is held fixed.

Passing `verbose = true` provides verbose output of the iterations.
"""
function pirls!(
    m::GeneralizedLinearMixedModel{T},
    varyβ=false,
    verbose=false;
    maxiter::Integer=m.LMM.optsum.pirls_maxiter,
    maxhalfstep::Integer=m.LMM.optsum.pirls_maxhalfstep,
    ftol_rel::Real=m.LMM.optsum.pirls_ftol_rel,
    ftol_abs::Real=m.LMM.optsum.pirls_ftol_abs,
) where {T}
    u₀ = m.u₀
    u = m.u
    β = m.β
    β₀ = m.β₀
    lm = m.LMM
    for j in eachindex(u)         # start from u all zeros
        copyto!(u₀[j], fill!(u[j], 0))
    end
    if varyβ
        copyto!(β₀, β)
        Llast = last(lm.L)
        pp1 = size(Llast, 1)
        Ltru = view(Llast, pp1, 1:(pp1 - 1)) # name read as L'u
    end
    obj₀ = deviance!(m) * 1.0001
    if verbose
        print("varyβ = ", varyβ, ", obj₀ = ", obj₀)
        if varyβ
            print(", β = ")
            show(β)
        end
        println()
    end
    for iter in 1:maxiter
        varyβ && ldiv!(adjoint(feL(m)), copyto!(β, Ltru))
        ranef!(u, m.LMM, β, true) # solve for new values of u
        obj = deviance!(m)        # update GLM vecs and evaluate Laplace approx
        verbose && println(lpad(iter, 4), ": ", obj)
        nhalf = 0
        while obj > obj₀
            nhalf += 1
            if nhalf > maxhalfstep
                if iter < 2
                    throw(ErrorException("number of averaging steps > $maxhalfstep"))
                end
                break
            end
            for i in eachindex(u)
                map!(average, u[i], u[i], u₀[i])
            end
            varyβ && map!(average, β, β, β₀)
            obj = deviance!(m)
            verbose && println(lpad(nhalf, 8), ", ", obj)
        end
        if isapprox(obj, obj₀; rtol=ftol_rel, atol=ftol_abs)
            break
        end
        copyto!.(u₀, u)
        copyto!(β₀, β)
        obj₀ = obj
    end
    return m
end

ranef(m::GeneralizedLinearMixedModel; uscale::Bool=false) = ranef(m.LMM; uscale=uscale)

LinearAlgebra.rank(m::GeneralizedLinearMixedModel) = m.LMM.feterm.rank

"""
    refit!(m::GeneralizedLinearMixedModel[, y::Vector];
           fast::Bool = (length(m.θ) == length(m.optsum.final)),
           nAGQ::Integer = m.optsum.nAGQ,
           kwargs...)

Refit the model `m` after installing response `y`.

If `y` is omitted the current response vector is used.

If not specified, the `fast` and `nAGQ` options from the previous fit are used.
`kwargs` are the same as [`fit!`](@ref)
"""
function refit!(
    m::GeneralizedLinearMixedModel;
    fast::Bool=(length(m.θ) == length(m.optsum.final)),
    nAGQ::Integer=m.optsum.nAGQ,
    kwargs...,
)
    return fit!(unfit!(m); fast=fast, nAGQ=nAGQ, kwargs...)
end

function refit!(m::GeneralizedLinearMixedModel, y; kwargs...)
    m_resp_y = m.resp.y
    length(y) == size(m_resp_y, 1) || throw(DimensionMismatch(""))
    copyto!(m_resp_y, y)
    return refit!(m; kwargs...)
end

"""
    setβθ!(m::GeneralizedLinearMixedModel, v)

Set the parameter vector, `:βθ`, of `m` to `v`.

`βθ` is the concatenation of the fixed-effects, `β`, and the covariance parameter, `θ`.

When ϕ is a free parameter of the outer optimization (`!isempty(m.ϕ)`, see
[`_dispersion`](@ref)) `v` carries one further element, `log(ϕ)`, after `θ`.
"""
function setβθ!(m::GeneralizedLinearMixedModel, v)
    setβ!(m, v)
    nβ = length(m.β)
    # `v` is allowed to be shorter than `nβ + nθ`: `simulate!` passes β alone
    # when θ is to be left at its current value, and relies on the resulting
    # empty view making `setθ!` a no-op.
    nϕ = length(m.ϕ)
    setθ!(m, view(v, (nβ + 1):(length(v) - nϕ)))
    # when ϕ is a free outer parameter it is the trailing element of `v`, on the
    # log scale so that the optimizer sees it as unbounded and reasonably scaled
    iszero(nϕ) || (m.ϕ[begin] = exp(last(v)))
    return m
end

function setβ!(m::GeneralizedLinearMixedModel, v)
    β = m.β
    copyto!(β, view(v, 1:length(β)))
    return m
end

function setθ!(m::GeneralizedLinearMixedModel, v)
    setθ!(m.LMM, copyto!(m.θ, v))
    return m
end

function Base.setproperty!(m::GeneralizedLinearMixedModel, s::Symbol, y)
    if s == :β
        setβ!(m, y)
    elseif s == :θ
        setθ!(m, y)
    elseif s == :βθ
        setβθ!(m, y)
    else
        setfield!(m, s, y)
    end
end

"""
    sdest(m::GeneralizedLinearMixedModel)

Return the estimate of the dispersion, i.e. the standard deviation of the per-observation noise.

For models with a dispersion parameter ϕ, this is simply ϕ. For models without a
dispersion parameter, this value is `missing`. This differs from `disperion`,
which returns `1` for models without a dispersion parameter.

For Gaussian models, this parameter is often called σ.
"""
function sdest(m::GeneralizedLinearMixedModel{T}) where {T}
    return dispersion_parameter(m) ? dispersion(m, false) : missing
end

function Base.show(
    io::IO, ::MIME"text/plain", m::GeneralizedLinearMixedModel{T,D}
) where {T,D}
    if m.optsum.feval < 0
        @warn("Model has not been fit")
        return nothing
    end
    nAGQ = m.LMM.optsum.nAGQ
    println(io, "Generalized Linear Mixed Model fit by maximum likelihood (nAGQ = $nAGQ)")
    println(io, "  ", m.LMM.formula)
    println(io, "  Distribution: ", D)
    println(io, "  Link: ", Link(m), "\n")
    nums = Ryu.writefixed.([loglikelihood(m), deviance(m), aic(m), aicc(m), bic(m)], 4)
    fieldwd = max(maximum(textwidth.(nums)) + 1, 11)
    for label in [" logLik", " deviance", "AIC", "AICc", "BIC"]
        print(io, rpad(lpad(label, (fieldwd + textwidth(label)) >> 1), fieldwd))
    end
    println(io)
    print.(Ref(io), lpad.(nums, fieldwd))
    println(io)
    println(io)

    show(io, VarCorr(m))

    print(io, " Number of obs: $(length(m.y)); levels of grouping factors: ")
    join(io, nlevs.(m.reterms), ", ")
    println(io)

    println(io, "\nFixed-effects parameters:")
    return show(io, MIME("text/plain"), coeftable(m))
end

Base.show(io::IO, m::GeneralizedLinearMixedModel) = show(io, MIME("text/plain"), m)

function stderror!(v::AbstractVector{T}, m::GeneralizedLinearMixedModel{T}) where {T}
    # initialize to appropriate NaN for rank-deficient case
    fill!(v, zero(T) / zero(T))

    # the inverse permutation is done here.
    # if this is changed to access the permuted
    # model matrix directly, then don't forget to add
    # in the inverse permutation
    vcovmat = vcov(m)

    for idx in 1:size(vcovmat, 1)
        v[idx] = sqrt(vcovmat[idx, idx])
    end

    return v
end

function unfit!(model::GeneralizedLinearMixedModel{T}) where {T}
    reevaluateAend!(model.LMM)

    reterms = model.LMM.reterms
    optsum = model.LMM.optsum
    optsum.initial = map(x -> T(x[2] == x[3]), model.LMM.parmap)
    optsum.final = copy(optsum.initial)
    optsum.xtol_abs = fill!(copy(optsum.initial), 1.0e-10)
    optsum.initial_step = T[]
    optsum.feval = -1
    empty!(model.ϕ)   # back to the plug-in until `fit!` decides otherwise
    deviance!(model, 1)

    return model
end

"""
    updateη!(m::GeneralizedLinearMixedModel)

Update the linear predictor, `m.η`, from the offset and the `B`-scale random effects.
"""
function updateη!(m::GeneralizedLinearMixedModel{T}) where {T}
    η = m.η
    b = m.b
    u = m.u
    reterms = m.LMM.reterms
    mul!(η, fullrankx(m), m.β)
    for i in eachindex(b)
        mul!(η, reterms[i], vec(mul!(b[i], reterms[i].λ, u[i])), one(T), one(T))
    end
    GLM.updateμ!(m.resp, η)
    return m
end

"""
    varest(m::GeneralizedLinearMixedModel)

Returns the estimate of ϕ², the variance of the conditional distribution of Y given B.

For models with a dispersion parameter ϕ, this is simply ϕ². For models without a
dispersion parameter, this value is `missing`. This differs from `disperion`,
which returns `1` for models without a dispersion parameter.

For Gaussian models, this parameter is often called σ².
"""
function varest(m::GeneralizedLinearMixedModel{T}) where {T}
    return dispersion_parameter(m) ? dispersion(m, true) : missing
end

function StatsAPI.weights(m::GeneralizedLinearMixedModel{T}) where {T}
    wts = m.wt
    return isempty(wts) ? ones(T, nobs(m)) : wts
end

# delegate GLMM method to LMM field
for f in (:feL, :fetrm, :fixefnames, :(LinearAlgebra.logdet), :PCA, :rePCA)
    @eval begin
        $f(m::GeneralizedLinearMixedModel) = $f(m.LMM)
    end
end
