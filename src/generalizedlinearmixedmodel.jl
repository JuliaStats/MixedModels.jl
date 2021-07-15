"""
    GeneralizedLinearMixedModel

Generalized linear mixed-effects model representation

# Fields
- `LMM`: a [`LinearMixedModel`](@ref) - the local approximation to the GLMM.
- `β`: the pivoted and possibly truncated fixed-effects vector
- `β₀`: similar to `β`. Used in the PIRLS algorithm if step-halving is needed.
- `θ`: covariance parameter vector
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
- `lowerbd`: vector of lower bounds on the combined elements of `β` and `θ`
- `formula`, `trms`, `A`, `L`, and `optsum`: fields of the `LMM` field
- `X`: fixed-effects model matrix
- `y`: response vector

"""
struct GeneralizedLinearMixedModel{T<:AbstractFloat, D<:Distribution} <: MixedModel{T}
    LMM::LinearMixedModel{T}
    β::Vector{T}
    β₀::Vector{T}
    θ::Vector{T}
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

function StatsBase.coef(m::GeneralizedLinearMixedModel{T}) where {T}
    piv = m.LMM.feterm.piv
    invpermute!(copyto!(fill(T(-0.0), length(piv)), m.β), piv)
end

function StatsBase.coeftable(m::GeneralizedLinearMixedModel)
    co = coef(m)
    se = stderror(m)
    z = co ./ se
    pvalue = ccdf.(Chisq(1), abs2.(z))
    CoefTable(
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
"""
function StatsBase.deviance(m::GeneralizedLinearMixedModel{T}, nAGQ = 1) where {T}
    nAGQ == 1 && return T(sum(m.resp.devresid) + logdet(m) + sum(u -> sum(abs2, u), m.u))
    u = vec(first(m.u))
    u₀ = vec(first(m.u₀))
    copyto!(u₀, u)
    ra = RaggedArray(m.resp.devresid, first(m.LMM.reterms).refs)
    devc0 = sum!(map!(abs2, m.devc0, u), ra)  # the deviance components at z = 0
    sd = map!(inv, m.sd, first(m.LMM.L).diag)
    mult = fill!(m.mult, 0)
    devc = m.devc
    for (z, w) in GHnorm(nAGQ)
        if !iszero(w)
            if iszero(z)  # devc == devc0 in this case
                mult .+= w
            else
                @. u = u₀ + z * sd
                updateη!(m)
                sum!(map!(abs2, devc, u), ra)
                @. mult += exp((abs2(z) + devc0 - devc) / 2) * w
            end
        end
    end
    copyto!(u, u₀)
    updateη!(m)
    sum(devc0) - 2 * (sum(log, mult) + sum(log, sd))
end

StatsBase.deviance(m::GeneralizedLinearMixedModel) = deviance(m, m.optsum.nAGQ)

fixef(m::GeneralizedLinearMixedModel) = m.β

function fixef!(v::AbstractVector{T}, m::GeneralizedLinearMixedModel{T}) where T
    copyto!(fill!(v, -zero(T)), m.β)
end

objective(m::GeneralizedLinearMixedModel) = deviance(m)

"""
    GLM.wrkresp!(v::AbstractVector{T}, resp::GLM.GlmResp{AbstractVector{T}})

A copy of a method from GLM that generalizes the types in the signature
"""
function GLM.wrkresp!(v::AbstractVector{T}, r::GLM.GlmResp{Vector{T}}) where {T<:AbstractFloat}
    v .= r.eta .+ r.wrkresid
    isempty(r.offset) && return v
    v .-= r.offset
end

"""
    deviance!(m::GeneralizedLinearMixedModel, nAGQ=1)

Update `m.η`, `m.μ`, etc., install the working response and working weights in
`m.LMM`, update `m.LMM.A` and `m.LMM.R`, then evaluate the `deviance`.
"""
function deviance!(m::GeneralizedLinearMixedModel, nAGQ = 1)
    updateη!(m)
    GLM.wrkresp!(m.LMM.y, m.resp)
    reweight!(m.LMM, m.resp.wrkwt)
    deviance(m, nAGQ)
end

function GLM.dispersion(m::GeneralizedLinearMixedModel{T}, sqr::Bool = false) where {T}
# adapted from GLM.dispersion(::AbstractGLM, ::Bool)
# TODO: PR for a GLM.dispersion(resp::GLM.GlmResp, dof_residual::Int, sqr::Bool)
    r = m.resp
    if dispersion_parameter(r.d)
        s = sum(wt * abs2(re) for (wt, re) in zip(r.wrkwt, r.wrkresid)) / dof_residual(m)
        sqr ? s : sqrt(s)
    else
        one(T)
    end
end

GLM.dispersion_parameter(m::GeneralizedLinearMixedModel) = dispersion_parameter(m.resp.d)

Distributions.Distribution(m::GeneralizedLinearMixedModel{T,D}) where {T,D} = D

fit(
    ::Type{GeneralizedLinearMixedModel},
    f::FormulaTerm,
    tbl,
    d::Distribution = Normal(),
    l::Link = canonicallink(d);
    wts = [],
    contrasts = Dict{Symbol,Any}(),
    offset = [],
    verbose::Bool = false,
    fast::Bool = false,
    nAGQ::Integer = 1,
    progress::Bool = true,
) = fit(
    GeneralizedLinearMixedModel,
    f,
    columntable(tbl),
    d,
    l,
    wts = wts,
    offset = offset,
    contrasts = contrasts,
    verbose = verbose,
    fast = fast,
    nAGQ = nAGQ,
    progress = progress,
)

fit(
    ::Type{GeneralizedLinearMixedModel},
    f::FormulaTerm,
    tbl::Tables.ColumnTable,
    d::Distribution,
    l::Link = canonicallink(d);
    wts = [],
    contrasts = Dict{Symbol,Any}(),
    offset = [],
    verbose::Bool = false,
    fast::Bool = false,
    nAGQ::Integer = 1,
    progress::Bool = true,
) = fit!(
    GeneralizedLinearMixedModel(
        f,
        tbl,
        d,
        l,
        wts = wts,
        offset = offset,
        contrasts = contrasts,
    ),
    verbose = verbose,
    fast = fast,
    nAGQ = nAGQ,
    progress = progress,
)


fit(
    ::Type{MixedModel},
    f::FormulaTerm,
    tbl,
    d::Distribution,
    l::Link = canonicallink(d);
    wts = [],
    contrasts = Dict{Symbol,Any}(),
    offset = [],
    verbose::Bool = false,
    REML::Bool = false,
    fast::Bool = false,
    nAGQ::Integer = 1,
    progress::Bool = true,
) = fit(
    GeneralizedLinearMixedModel,
    f,
    tbl,
    d,
    l,
    wts = wts,
    contrasts = contrasts,
    offset = offset,
    verbose = verbose,
    fast = fast,
    nAGQ = nAGQ,
    progress = progress,
)

"""
    fit!(m::GeneralizedLinearMixedModel[, verbose=false, fast=false, nAGQ=1, progress=true])

Optimize the objective function for `m`.

When `fast` is `true` a potentially much faster but slightly less accurate algorithm, in
which `pirls!` optimizes both the random effects and the fixed-effects parameters,
is used.

If `progress` is `true`, the default, a `ProgressMeter.ProgressUnknown` counter is displayed.
during the iterations to minimize the deviance.  There is a delay before this display is initialized
and it may not be shown at all for models that are optimized quickly.

If `verbose` is `true`, then both the intermediate results of both the nonlinear optimization and PIRLS are also displayed on standard output.
"""
function fit!(
    m::GeneralizedLinearMixedModel{T};
    verbose::Bool = false,
    fast::Bool = false,
    nAGQ::Integer = 1,
    progress::Bool = true,
) where {T}
    β = m.β
    lm = m.LMM
    optsum = lm.optsum

    if optsum.feval > 0
        throw(ArgumentError("This model has already been fitted. Use refit!() instead."))
    end

    if !fast
        optsum.lowerbd = vcat(fill!(similar(β), T(-Inf)), optsum.lowerbd)
        optsum.initial = vcat(β, m.θ)
        optsum.final = copy(optsum.initial)
    end
    setpar! = fast ? setθ! : setβθ!
    prog = ProgressUnknown("Minimizing"; showspeed=true)
    function obj(x, g)
        isempty(g) || throw(ArgumentError("g should be empty for this objective"))
        val = deviance(pirls!(setpar!(m, x), fast, verbose), nAGQ)
        verbose && println(round(val, digits = 5), " ", x)
        progress && ProgressMeter.next!(prog; showvalues = [(:objective, val),])
        val
    end
    opt = Opt(optsum)
    NLopt.min_objective!(opt, obj)
    optsum.finitial = obj(optsum.initial, T[])
    fmin, xmin, ret = NLopt.optimize(opt, copyto!(optsum.final, optsum.initial))
    ProgressMeter.finish!(prog)
    ## check if very small parameter values bounded below by zero can be set to zero
    xmin_ = copy(xmin)
    for i in eachindex(xmin_)
        if iszero(optsum.lowerbd[i]) && zero(T) < xmin_[i] < T(0.001)
            xmin_[i] = zero(T)
        end
    end
    if xmin ≠ xmin_
        if (zeroobj = obj(xmin_, T[])) ≤ (fmin + 1.e-5)
            fmin = zeroobj
            copyto!(xmin, xmin_)
        end
    end
    ## ensure that the parameter values saved in m are xmin
    pirls!(setpar!(m, xmin), fast, verbose)
    optsum.nAGQ = nAGQ
    optsum.feval = opt.numevals
    optsum.final = xmin
    optsum.fmin = fmin
    optsum.returnvalue = ret
    _check_nlopt_return(ret)
    m
end

StatsBase.fitted(m::GeneralizedLinearMixedModel) = m.resp.mu

GeneralizedLinearMixedModel(
    f::FormulaTerm,
    tbl,
    d::Distribution,
    l::Link = canonicallink(d);
    wts = [],
    offset = [],
    contrasts = Dict{Symbol,Any}(),
) = GeneralizedLinearMixedModel(
    f,
    Tables.columntable(tbl),
    d,
    l;
    wts = wts,
    offset = offset,
    contrasts = contrasts)

GeneralizedLinearMixedModel(
    f::FormulaTerm,
    tbl::Tables.ColumnTable,
    d::Normal,
    l::IdentityLink;
    wts = [],
    offset = [],
    contrasts = Dict{Symbol,Any}(),
) = throw(ArgumentError("use LinearMixedModel for Normal distribution with IdentityLink"))

function GeneralizedLinearMixedModel(
    f::FormulaTerm,
    tbl::Tables.ColumnTable,
    d::Distribution,
    l::Link = canonicallink(d);
    wts = [],
    offset = [],
    contrasts = Dict{Symbol,Any}(),
)
    if isa(d, Binomial) && isempty(wts)
        d = Bernoulli()
    end
    (isa(d, Normal) && isa(l, IdentityLink)) &&
    throw(ArgumentError("use LinearMixedModel for Normal distribution with IdentityLink"))

    if !any(isa(d, dist) for dist in (Bernoulli, Binomial, Poisson))
        @warn """Results for families with a dispersion parameter are not reliable.
                 It is best to avoid trying to fit such models in MixedModels until
                 the authors gain a better understanding of those cases."""
    end

    LMM = LinearMixedModel(f, tbl, contrasts = contrasts; wts = wts)
    y = copy(LMM.y)
        # the sqrtwts field must be the correct length and type but we don't know those
        # until after the model is constructed if wt is empty.  Because a LinearMixedModel
        # type is immutable, another one must be created.
    if isempty(wts)
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
    updateL!(LMM)
        # fit a glm to the fixed-effects only
    T = eltype(LMM.Xymat)
    gl = glm(LMM.X, y, d, l, wts=convert(Vector{T}, wts), offset=convert(Vector{T}, offset))
    β = coef(gl)
    u = [fill(zero(eltype(y)), vsize(t), nlevs(t)) for t in LMM.reterms]
        # vv is a template vector used to initialize fields for AGQ
        # it is empty unless there is a single random-effects term
    vv = length(u) == 1 ? vec(first(u)) : similar(y, 0)

    res = GeneralizedLinearMixedModel{T, typeof(d)}(
        LMM,
        β,
        copy(β),
        LMM.θ,
        copy.(u),
        u,
        zero.(u),
        gl.rr,
        similar(y),
        oftype(y, wts),
        similar(vv),
        similar(vv),
        similar(vv),
        similar(vv),
    )
    deviance!(res, 1)
    res
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
    elseif s ∈ (:A, :L, :optsum, :reterms, :Xymat, :feterm, :formula, :parmap)
        getfield(m.LMM, s)
    elseif s ∈ (:dims, :λ, :lowerbd, :corr, :PCA, :rePCA, :X,)
        getproperty(m.LMM, s)
    elseif s == :y
        m.resp.y
    else
        getfield(m, s)
    end
end

# this copy behavior matches the implicit copy behavior
# for LinearMixedModel. So this is then different than m.θ,
# which returns a reference to the same array
getθ(m::GeneralizedLinearMixedModel)  = copy(m.θ)
getθ!(v::AbstractVector{T}, m::GeneralizedLinearMixedModel{T}) where {T} = copyto!(v, m.θ)

StatsBase.islinear(m::GeneralizedLinearMixedModel) = isa(GLM.Link, GLM.IdentityLink)

GLM.Link(m::GeneralizedLinearMixedModel) = GLM.Link(m.resp)

function StatsBase.loglikelihood(m::GeneralizedLinearMixedModel{T}) where {T}
    accum = zero(T)
    # adapted from GLM.jl
    # note the use of loglik_obs to handle the different parameterizations
    # of various response distributions which may not just be location+scale
    r   = m.resp
    wts = r.wts
    y   = r.y
    mu  = r.mu
    d   = r.d
    if length(wts) == length(y)
        ϕ = deviance(r)/sum(wts)
        @inbounds for i in eachindex(y, mu, wts)
            accum += GLM.loglik_obs(d, y[i], mu[i], wts[i], ϕ)
        end
    else
        ϕ = deviance(r)/length(y)
        @inbounds for i in eachindex(y, mu)
            accum += GLM.loglik_obs(d, y[i], mu[i], 1, ϕ)
        end
    end
    accum  - (mapreduce(u -> sum(abs2, u), +, m.u) + logdet(m)) / 2
end

Base.propertynames(m::GeneralizedLinearMixedModel, private::Bool = false) = (
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
    :lowerbd,
    :objective,
    :σρs,
    :σs,
    :corr,
    :vcov,
    :PCA,
    :rePCA,
    (private ? fieldnames(GeneralizedLinearMixedModel) : (:LMM, :β, :θ, :b, :u, :resp, :wt))...,
)

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
    varyβ = false,
    verbose = false;
    maxiter::Integer = 10,
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
    for iter = 1:maxiter
        varyβ && ldiv!(adjoint(feL(m)), copyto!(β, Ltru))
        ranef!(u, m.LMM, β, true) # solve for new values of u
        obj = deviance!(m)        # update GLM vecs and evaluate Laplace approx
        verbose && println(lpad(iter, 4), ": ", obj)
        nhalf = 0
        while obj > obj₀
            nhalf += 1
            if nhalf > 10
                if iter < 2
                    throw(ErrorException("number of averaging steps > 10"))
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
        if isapprox(obj, obj₀; atol = 0.00001)
            break
        end
        copyto!.(u₀, u)
        copyto!(β₀, β)
        obj₀ = obj
    end
    m
end

ranef(m::GeneralizedLinearMixedModel; uscale::Bool=false) = ranef(m.LMM, uscale=uscale)

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
function refit!(m::GeneralizedLinearMixedModel;
                fast::Bool = (length(m.θ) == length(m.optsum.final)),
                nAGQ::Integer = m.optsum.nAGQ, kwargs...)
    fit!(unfit!(m); fast=fast, nAGQ=nAGQ, kwargs...)
end

function refit!(m::GeneralizedLinearMixedModel, y; kwargs...)
    m_resp_y = m.resp.y
    length(y) == size(m_resp_y, 1) || throw(DimensionMismatch(""))
    copyto!(m_resp_y, y)
    refit!(m; kwargs...)
end

"""
    setβθ!(m::GeneralizedLinearMixedModel, v)

Set the parameter vector, `:βθ`, of `m` to `v`.

`βθ` is the concatenation of the fixed-effects, `β`, and the covariance parameter, `θ`.
"""
function setβθ!(m::GeneralizedLinearMixedModel, v)
    setβ!(m, v)
    setθ!(m, view(v, (length(m.β)+1):length(v)))
end

function setβ!(m::GeneralizedLinearMixedModel, v)
    β = m.β
    copyto!(β, view(v, 1:length(β)))
    m
end

function setθ!(m::GeneralizedLinearMixedModel, v)
    setθ!(m.LMM, copyto!(m.θ, v))
    m
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
sdest(m::GeneralizedLinearMixedModel{T}) where {T} =  dispersion_parameter(m) ? dispersion(m, false) : missing

function Base.show(io::IO, ::MIME"text/plain", m::GeneralizedLinearMixedModel{T,D}) where {T,D}
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
    show(io, coeftable(m))
end

Base.show(io::IO,  m::GeneralizedLinearMixedModel) = show(io, MIME"text/plain"(), m)

function stderror!(v::AbstractVector{T}, m::GeneralizedLinearMixedModel{T}) where {T}
    # initialize to appropriate NaN for rank-deficient case
    fill!(v, zero(T) / zero(T))

    # the inverse permutation is done here.
    # if this is changed to access the permuted
    # model matrix directly, then don't forget to add
    # in the inverse permutation
    vcovmat = vcov(m)

    for idx in 1:size(vcovmat,1)
        v[idx] = sqrt(vcovmat[idx,idx])
    end

    v
end

function unfit!(model::GeneralizedLinearMixedModel{T}) where {T}
    deviance!(model, 1)
    reevaluateAend!(model.LMM)

    reterms = model.LMM.reterms
    optsum = model.LMM.optsum
    # we need to reset optsum so that it
    # plays nice with the modifications fit!() does
    optsum.lowerbd = mapfoldl(lowerbd, vcat, reterms)
    optsum.initial = mapfoldl(getθ, vcat, reterms)
    optsum.final = copy(optsum.initial)
    optsum.xtol_abs = fill!(copy(optsum.initial), 1.0e-10)
    optsum.initial_step = T[]
    optsum.feval = -1

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
    mul!(η, modelmatrix(m), m.β)
    for i in eachindex(b)
        mul!(η, reterms[i], vec(mul!(b[i], reterms[i].λ, u[i])), one(T), one(T))
    end
    GLM.updateμ!(m.resp, η)
    m
end

"""
    varest(m::GeneralizedLinearMixedModel)

Returns the estimate of ϕ², the variance of the conditional distribution of Y given B.

For models with a dispersion parameter ϕ, this is simply ϕ². For models without a
dispersion parameter, this value is `missing`. This differs from `disperion`,
which returns `1` for models without a dispersion parameter.

For Gaussian models, this parameter is often called σ².
"""
varest(m::GeneralizedLinearMixedModel{T}) where {T} = dispersion_parameter(m) ? dispersion(m, true) : missing

function StatsBase.weights(m::GeneralizedLinearMixedModel{T}) where {T}
    wts = m.wt
    isempty(wts) ? ones(T, nobs(m)) : wts
end

# delegate GLMM method to LMM field
for f in (
    :feL,
    :fetrm,
    :fixefnames,
    :(LinearAlgebra.logdet),
    :lowerbd,
    :PCA,
    :rePCA,
)
    @eval begin
        $f(m::GeneralizedLinearMixedModel) = $f(m.LMM)
    end
end
