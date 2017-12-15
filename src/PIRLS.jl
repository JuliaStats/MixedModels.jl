function StatsBase.dof(m::GeneralizedLinearMixedModel)
    length(m.β) + length(m.θ) + GLM.dispersion_parameter(m.resp.d)
end

StatsBase.fitted(m::GeneralizedLinearMixedModel) = m.resp.mu

function fixef(m::GeneralizedLinearMixedModel{T}, permuted=true) where T
    permuted && return m.β
    Xtrm = m.LMM.trms[end - 1]
    iperm = invperm(Xtrm.piv)
    p = length(iperm)
    r = Xtrm.rank
    r == p ? m.β[iperm] : copy!(fill(-zero(T), p), m.β)[iperm]
end

"""
    glmm(f::Formula, fr::ModelFrame, d::Distribution[, l::GLM.Link])

Return a `GeneralizedLinearMixedModel` object.

The value is ready to be `fit!` but has not yet been fit.
"""
function glmm(f::Formula, fr::AbstractDataFrame, d::Distribution, l::Link; wt=[], offset=[],
              contrasts = Dict())
    if d == Binomial() && isempty(wt)
        d = Bernoulli()
    end
    LMM = lmm(f, fr; weights = wt, contrasts=contrasts, rdist=d)
    X = LMM.trms[end - 1].x
    T = eltype(X)
    y = copy(model_response(LMM))
    if isempty(wt)
        LMM = LinearMixedModel(LMM.formula, LMM.trms, ones(y), LMM.A, LMM.L, LMM.optsum)
    end
    updateL!(setθ!(LMM, getθ(LMM)))
            # fit a glm to the fixed-effects only - awkward syntax is to by-pass a test
    gl = isempty(wt) ? glm(X, y, d, l) : glm(X, y, d, l, wts=wt)
    β = coef(gl)
    u = [zeros(T, vsize(t), nlevs(t)) for t in reterms(LMM)]
    res = GeneralizedLinearMixedModel(LMM, β, copy(β), getθ(LMM), copy.(u), u,
        zeros.(u), gl.rr, similar(y), oftype(y, wt))
    setβθ!(res, vcat(coef(gl), getθ(LMM)))
    LaplaceDeviance!(res)
    res
end

glmm(f::Formula, fr::AbstractDataFrame, d::Distribution; wt=[], offset=[], contrasts=Dict()) =
    glmm(f, fr, d, GLM.canonicallink(d), wt=wt, offset=offset, contrasts=contrasts)

"""
    LaplaceDeviance(m::GeneralizedLinearMixedModel{T})::T where T

Return the Laplace approximation to the deviance of `m`.

If the distribution `D` does not have a scale parameter the Laplace approximation
is defined as the squared length of the conditional modes, `u`, plus the determinant
of `Λ'Z'WZΛ + I`, plus the sum of the squared deviance residuals.
"""
LaplaceDeviance(m::GeneralizedLinearMixedModel{T}) where T =
    T(sum(m.resp.devresid) + logdet(m) + sum(u -> sum(abs2, u), m.u))

"""
    LaplaceDeviance!(m::GeneralizedLinearMixedModel)

Update `m.η`, `m.μ`, etc., install the working response and working weights in
`m.LMM`, update `m.LMM.A` and `m.LMM.R`, then evaluate `LaplaceDeviance`.
"""
function LaplaceDeviance!(m::GeneralizedLinearMixedModel)
    updateη!(m)
    GLM.wrkresp!(vec(m.LMM.trms[end].x), m.resp)
    reweight!(m.LMM, m.resp.wrkwt)
    LaplaceDeviance(m)
end

function loglikelihood(m::GeneralizedLinearMixedModel{T}) where T
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

function lowerbd(m::GeneralizedLinearMixedModel)
    lb = lowerbd(m.LMM)
    vcat(fill(convert(eltype(lb), -Inf), size(m.β)), lb)
end

StatsBase.nobs(m::GeneralizedLinearMixedModel) = length(m.η)

StatsBase.predict(m::GeneralizedLinearMixedModel) = fitted(m)

function StatsBase.predict(m::GeneralizedLinearMixedModel{T},df::AbstractDataFrame) where T
    linkinv(Link(m.resp),StatsBase.predict(m.LMM,df))
end

"""
    updateη!(m::GeneralizedLinearMixedModel)

Update the linear predictor, `m.η`, from the offset and the `B`-scale random effects.
"""
function updateη!(m::GeneralizedLinearMixedModel)
    η = m.η
    b = m.b
    u = m.u
    trms = m.LMM.trms
    A_mul_B!(η, trms[end - 1].x, m.β)
    for i in eachindex(b)
        unscaledre!(η, trms[i], Λ_mul_B!(b[i], trms[i], u[i]))
    end
    updateμ!(m.resp, η)
    m
end

average(a::T, b::T) where {T <: AbstractFloat} = (a + b) / 2

"""
    pirls!(m::GeneralizedLinearMixedModel)

Use Penalized Iteratively Reweighted Least Squares (PIRLS) to determine the conditional
modes of the random effects.

When `varyβ` is true both `u` and `β` are optimized with PIRLS.  Otherwise only `u` is
optimized and `β` is held fixed.

Passing `verbose = true` provides verbose output of the iterations.
"""
function pirls!(m::GeneralizedLinearMixedModel{T}, varyβ::Bool=false, verbose::Bool=false) where T
    iter, maxiter, obj = 0, 100, T(-Inf)
    u₀ = m.u₀
    u = m.u
    β = m.β
    β₀ = m.β₀
    lm = m.LMM
    for j in eachindex(u)         # start from u all zeros
        copy!(u₀[j], fill!(u[j], 0))
    end
    varyβ && copy!(β₀, β)
    obj₀ = LaplaceDeviance!(m) * 1.0001
    verbose && @show(varyβ, obj₀, β)

    while iter < maxiter
        iter += 1
        varyβ && Ac_ldiv_B!(feL(m), copy!(β, lm.L.data.blocks[end, end - 1]))
        ranef!(u, m.LMM, β, true) # solve for new values of u
        obj = LaplaceDeviance!(m) # update GLM vecs and evaluate Laplace approx
        verbose && @show(iter, obj)
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
            obj = LaplaceDeviance!(m)
            verbose && @show(nhalf, obj)
        end
        if isapprox(obj, obj₀; atol = 0.00001)
            break
        end
        copy!.(u₀, u)
        copy!(β₀, β)
        obj₀ = obj
    end
    obj
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
    copy!(β, view(v, 1 : length(β)))
    m
end

function setθ!(m::GeneralizedLinearMixedModel, v)
    setθ!(m.LMM, copy!(m.θ, v))
    m
end

sdest(m::GeneralizedLinearMixedModel{T}) where T = convert(T, NaN)

"""
    fit!(m::GeneralizedLinearMixedModel[, verbose = false, fast = false])

Optimize the objective function for `m`.

When `fast` is `true` a potentially much faster but slightly less accurate algorithm, in
which `pirls!` optimizes both the random effects and the fixed-effects parameters,
is used.
"""
function StatsBase.fit!(m::GeneralizedLinearMixedModel{T};
                        verbose::Bool=false, fast::Bool=false) where T
    β = m.β
    lm = m.LMM
    optsum = lm.optsum
    if !fast
        fit!(m, verbose=verbose, fast=true)
        optsum.lowerbd = vcat(fill!(similar(β), T(-Inf)), optsum.lowerbd)
        optsum.initial = vcat(β, m.θ)
        optsum.final = copy(optsum.initial)
        optsum.initial_step = vcat(stderr(m) ./ 3, min.(T(0.05), m.θ ./ 4))
    end
    setpar! = fast ? setθ! : setβθ!
    feval = 0
    function obj(x, g)
        isempty(g) || error("gradient not defined for this model")
        feval += 1
        val = pirls!(setpar!(m, x), fast)
        feval == 1 && (optsum.finitial = val)
        verbose && println("f_", feval, ": ", round(val, 5), " ", x)
        val
    end
    opt = Opt(optsum)
    NLopt.min_objective!(opt, obj)
    fmin, xmin, ret = NLopt.optimize(opt, copy!(optsum.final, optsum.initial))
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
            copy!(xmin, xmin_)
        end
    end
    ## ensure that the parameter values saved in m are xmin
    pirls!(setpar!(m, xmin), fast)
    optsum.feval = feval
    optsum.final = xmin
    optsum.fmin = fmin
    optsum.returnvalue = ret
    ret == :ROUNDOFF_LIMITED && warn("NLopt was roundoff limited")
    if ret ∈ [:FAILURE, :INVALID_ARGS, :OUT_OF_MEMORY, :FORCED_STOP, :MAXEVAL_REACHED]
        warn("NLopt optimization failure: $ret")
    end
    m
end

function Base.show(io::IO, m::GeneralizedLinearMixedModel)
    println(io, "Generalized Linear Mixed Model fit by minimizing the Laplace approximation to the deviance")
    println(io, "  ", m.LMM.formula)
    println(io, "  Distribution: ", Distribution(m.resp))
    println(io, "  Link: ", Link(m.resp), "\n")
    println(io, string("  Deviance (Laplace approximation): ", @sprintf("%.4f", LaplaceDeviance(m))), "\n")

    show(io,VarCorr(m))
    gl = grplevels(m.LMM)
    print(io, "\n Number of obs: ", length(m.η), "; levels of grouping factors: ", gl[1])
    for l in gl[2:end]
        print(io, ", ", l)
    end
    println(io)
    println(io, "\nFixed-effects parameters:")
    show(io, coeftable(m))
end

varest(m::GeneralizedLinearMixedModel{T}) where T = one(T)
