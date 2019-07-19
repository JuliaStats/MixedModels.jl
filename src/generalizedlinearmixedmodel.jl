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
struct GeneralizedLinearMixedModel{T <: AbstractFloat} <: MixedModel{T}
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

"""
    deviance(m::GeneralizedLinearMixedModel{T}, nAGQ=1)::T where {T}

Return the deviance of `m` evaluated by the Laplace approximation (`nAGQ=1`)
or `nAGQ`-point adaptive Gauss-Hermite quadrature.

If the distribution `D` does not have a scale parameter the Laplace approximation
is the squared length of the conditional modes, `u`, plus the determinant
of `Λ'Z'WZΛ + I`, plus the sum of the squared deviance residuals.
"""
function StatsBase.deviance(m::GeneralizedLinearMixedModel{T}, nAGQ=1) where {T}
    nAGQ == 1 && return T(sum(m.resp.devresid) + logdet(m) + sum(u -> sum(abs2, u), m.u))
    u = vec(first(m.u))
    u₀ = vec(first(m.u₀))
    copyto!(u₀, u)
    ra = RaggedArray(m.resp.devresid, first(m.LMM.reterms).refs)
    devc0 = sum!(map!(abs2, m.devc0, u), ra)  # the deviance components at z = 0
    sd = map!(inv, m.sd, m.LMM.L[Block(1,1)].diag)
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
                @. mult += exp((abs2(z) + devc0 - devc)/2)*w
            end
        end
    end
    copyto!(u, u₀)
    updateη!(m)
    sum(devc0) - 2 * (sum(log, mult) + sum(log, sd))
end

"""
deviance!(m::GeneralizedLinearMixedModel, nAGQ=1)

Update `m.η`, `m.μ`, etc., install the working response and working weights in
`m.LMM`, update `m.LMM.A` and `m.LMM.R`, then evaluate the [`deviance`](@ref).
"""
function deviance!(m::GeneralizedLinearMixedModel, nAGQ=1)
    updateη!(m)
    GLM.wrkresp!(m.LMM.y, m.resp)
    reweight!(m.LMM, m.resp.wrkwt)
    deviance(m, nAGQ)
end

function StatsBase.dof(m::GeneralizedLinearMixedModel)
    length(m.β) + length(m.θ) + GLM.dispersion_parameter(m.resp.d)
end

StatsBase.fit(::Type{GeneralizedLinearMixedModel}, f::FormulaTerm, tbl, d::Distribution) =
    fit!(GeneralizedLinearMixedModel(f, columntable(tbl), d, GLM.canonicallink(d)))

StatsBase.fit(::Type{GeneralizedLinearMixedModel}, f::FormulaTerm, tbl, d::Distribution,
    l::GLM.Link) = fit!(GeneralizedLinearMixedModel(f,columntable(tbl), d, l))

"""
    fit!(m::GeneralizedLinearMixedModel[, verbose = false, fast = false, nAGQ=1])

Optimize the objective function for `m`.

When `fast` is `true` a potentially much faster but slightly less accurate algorithm, in
which `pirls!` optimizes both the random effects and the fixed-effects parameters,
is used.
"""
function StatsBase.fit!(m::GeneralizedLinearMixedModel{T};
                        verbose::Bool=false, fast::Bool=false, nAGQ=1) where {T}
    β = m.β
    lm = m.LMM
    optsum = lm.optsum
    if !fast
        fit!(m, verbose=verbose, fast=true, nAGQ=nAGQ)
        optsum.lowerbd = vcat(fill!(similar(β), T(-Inf)), optsum.lowerbd)
        optsum.initial = vcat(β, m.θ)
        optsum.final = copy(optsum.initial)
        optsum.initial_step = vcat(stderror(m) ./ 3, min.(T(0.05), m.θ ./ 4))
    end
    setpar! = fast ? setθ! : setβθ!
    feval = 0
    function obj(x, g)
        isempty(g) || error("gradient not defined for this model")
        feval += 1
        val = deviance(pirls!(setpar!(m, x), fast, verbose), nAGQ)
        feval == 1 && (optsum.finitial = val)
        if verbose
            println("f_", feval, ": ", val, " ", x)
        end
        val
    end
    opt = Opt(optsum)
    NLopt.min_objective!(opt, obj)
    fmin, xmin, ret = NLopt.optimize(opt, copyto!(optsum.final, optsum.initial))
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
    optsum.feval = feval
    optsum.final = xmin
    optsum.fmin = fmin
    optsum.returnvalue = ret
    ret == :ROUNDOFF_LIMITED && @warn("NLopt was roundoff limited")
    if ret ∈ [:FAILURE, :INVALID_ARGS, :OUT_OF_MEMORY, :FORCED_STOP, :MAXEVAL_REACHED]
        @warn("NLopt optimization failure: $ret")
    end
    m
end

StatsBase.fitted(m::GeneralizedLinearMixedModel) = m.resp.mu

function fixef(m::GeneralizedLinearMixedModel{T}, permuted=true) where {T}
    permuted && return m.β
    Xtrm = first(m.LMM.feterms)
    piv = Xtrm.piv
    v = fill(-zero(T), size(piv))
    copyto!(view(v, 1:Xtrm.rank), m.β)
    invpermute!(v, piv)
end

function GeneralizedLinearMixedModel(f::FormulaTerm, tbl::D, d::Distribution, l::GLM.Link;
        wt=[], offset=[], hints = Dict{Symbol,Any}()) where {D<:Tables.ColumnTable}
    if d == Binomial() && isempty(wt)
        d = Bernoulli()
    end
    LMM = LinearMixedModel(f, tbl, hints; weights = wt)
    y = copy(LMM.y)
        # the sqrtwts field must be the correct length and type but we don't know those
        # until after the model is constructed if wt is empty.  Because a LinearMixedModel
        # type is immutable, another one must be created.
    if isempty(wt)
        LMM = LinearMixedModel(LMM.formula, LMM.reterms, LMM.feterms, fill!(similar(y), 1), 
            LMM.A, LMM.L, LMM.optsum)
    end
    updateL!(LMM)
        # fit a glm to the fixed-effects only - awkward syntax is to by-pass a test
    gl = isempty(wt) ? glm(LMM.X, y, d, l) : glm(LMM.X, y, d, l, wts=wt)
    β = coef(gl)
    u = [fill(zero(eltype(y)), vsize(t), nlevs(t)) for t in LMM.reterms]
        # vv is a template vector used to initialize fields for AGQ
        # it is empty unless there is a single random-effects term
    vv = length(u) == 1 ? vec(first(u)) : similar(y, 0)
    
    res = GeneralizedLinearMixedModel(LMM, β, copy(β), LMM.θ, copy.(u), u,
        zero.(u), gl.rr, similar(y), oftype(y, wt), similar(vv),
        similar(vv), similar(vv), similar(vv))
    deviance!(res, 1)
    res
end

GeneralizedLinearMixedModel(f::FormulaTerm, tbl, d::Distribution;
        wt=[], offset=[], hints=Dict{Symbol,Any}()) =
    GeneralizedLinearMixedModel(f, columntable(tbl), d, GLM.canonicallink(d),
        wt=wt, offset=offset, hints=hints)

function Base.getproperty(m::GeneralizedLinearMixedModel, s::Symbol)
    if s == :theta
        m.θ
    elseif s == :coef
        coef(m)
    elseif s == :beta
        m.β 
    elseif s ∈ (:σ, :sigma)
        sdest(m)
    elseif s ∈ (:A, :L, :λ, :lowerbd, :optsum, :X, :reterms, :feterms, :formula)
        getproperty(m.LMM, s)
    elseif s == :y
        m.resp.y
    else
        getfield(m, s)
    end
end

function StatsBase.loglikelihood(m::GeneralizedLinearMixedModel{T}) where {T}
    accum = zero(T)
    D = Distribution(m.resp)
    if D <: Binomial
        for (μ, y, n) in zip(m.resp.mu, m.resp.y, m.wt)
            accum += logpdf(D(round(Int, n), μ), round(Int, y * n))
        end
    else
        for (μ, y) in zip(m.resp.mu, m.resp.y)
            accum += logpdf(D(μ), y)
        end
    end
    accum - (mapreduce(u -> sum(abs2, u), + , m.u) + logdet(m)) / 2
end

StatsBase.nobs(m::GeneralizedLinearMixedModel) = length(m.η)

StatsBase.predict(m::GeneralizedLinearMixedModel) = fitted(m)

Base.propertynames(m::GeneralizedLinearMixedModel, private=false) =
(:A, :L, :theta, :beta, :coef, :λ, :lambda, :σ, :sigma, :X, :y, :lowerbd, fieldnames(typeof(m))...)

"""
    pirls!(m::GeneralizedLinearMixedModel)

Use Penalized Iteratively Reweighted Least Squares (PIRLS) to determine the conditional
modes of the random effects.

When `varyβ` is true both `u` and `β` are optimized with PIRLS.  Otherwise only `u` is
optimized and `β` is held fixed.

Passing `verbose = true` provides verbose output of the iterations.
"""
function pirls!(m::GeneralizedLinearMixedModel{T}, varyβ=false, verbose=false; maxiter::Integer=10) where {T}
    u₀ = m.u₀
    u = m.u
    β = m.β
    β₀ = m.β₀
    lm = m.LMM
    for j in eachindex(u)         # start from u all zeros
        copyto!(u₀[j], fill!(u[j], 0))
    end
    varyβ && copyto!(β₀, β)
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
        varyβ && ldiv!(adjoint(feL(m)), copyto!(β, lm.L.blocks[end, end - 1]))
        ranef!(u, m.LMM, β, true) # solve for new values of u
        obj = deviance!(m)        # update GLM vecs and evaluate Laplace approx
        verbose && println(lpad(iter,4), ": ", obj)
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

"""
    setβθ!(m::GeneralizedLinearMixedModel, v)

Set the parameter vector, `:βθ`, of `m` to `v`.

`βθ` is the concatenation of the fixed-effects, `β`, and the covariance parameter, `θ`.
"""
function setβθ!(m::GeneralizedLinearMixedModel, v)
    setβ!(m, v)
    setθ!(m, view(v, (length(m.β) + 1) : length(v)))
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

sdest(m::GeneralizedLinearMixedModel{T}) where {T} = convert(T, NaN)

function Base.show(io::IO, m::GeneralizedLinearMixedModel)
    nAGQ = m.LMM.optsum.nAGQ
    println(io, "Generalized Linear Mixed Model fit by maximum likelihood (nAGQ = $nAGQ)")
    println(io, "  ", m.LMM.formula)
    println(io, "  Distribution: ", Distribution(m.resp))
    println(io, "  Link: ", GLM.Link(m.resp), "\n")
    println(io, string("  Deviance: ", @sprintf("%.4f", deviance(m, nAGQ))), "\n")

    show(io,VarCorr(m))

    print(io," Number of obs: $(length(m.y)); levels of grouping factors: ")
    join(io, nlevs.(m.reterms), ", ")
    println(io)

    println(io, "\nFixed-effects parameters:")
    show(io, coeftable(m))
end

"""
    updateη!(m::GeneralizedLinearMixedModel)

Update the linear predictor, `m.η`, from the offset and the `B`-scale random effects.
"""
function updateη!(m::GeneralizedLinearMixedModel)
    η = m.η
    b = m.b
    u = m.u
    reterms = m.LMM.reterms
    mul!(η, modelmatrix(m), m.β)
    for i in eachindex(b)
        unscaledre!(η, reterms[i], mul!(b[i], reterms[i].λ, u[i]))
    end
    GLM.updateμ!(m.resp, η)
    m
end

varest(m::GeneralizedLinearMixedModel{T}) where {T} = one(T)

            # delegate GLMM method to LMM field
for f in (
    :describeblocks,
    :feL, 
    :(LinearAlgebra.logdet), 
    :lowerbd,
    :(StatsModels.modelmatrix),
    :(StatsBase.vcov), 
    )
    @eval begin
        $f(m::GeneralizedLinearMixedModel) = $f(m.LMM)
    end
end

