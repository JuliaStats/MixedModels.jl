"""
    GeneralizedLinearMixedModel

Generalized linear mixed-effects model representation

Members:

- `LMM`: a [`LinearMixedModel`](@ref) - the local approximation to the GLMM.
- `β`: the fixed-effects vector
- `β₀`: similar to `β`. User in the PIRLS algorithm if step-halving is needed.
- `θ`: covariance parameter vector
- `b`: similar to `u`, equivalent to `broadcast!(*, b, LMM.Λ, u)`
- `u`: a vector of matrices of random effects
- `u₀`: similar to `u`.  Used in the PIRLS algorithm if step-halving is needed.
- `resp`: a `GlmResp` object
- `η`: the linear predictor
- `wt`: vector of prior case weights, a value of `T[]` indicates equal weights.
"""
struct GeneralizedLinearMixedModel{T <: AbstractFloat} <: MixedModel
    LMM::LinearMixedModel{T}
    β::Vector{T}
    β₀::Vector{T}
    θ::Vector{T}
    b::Vector{Matrix{T}}
    u::Vector{Matrix{T}}
    u₀::Vector{Matrix{T}}
    resp::GlmResp
    η::Vector{T}
    wt::Vector{T}
end

fixef(m::GeneralizedLinearMixedModel) = m.β

"""
    glmm(f::Formula, fr::ModelFrame, d::Distribution[, l::GLM.Link])

Return a `GeneralizedLinearMixedModel` object.

The value is ready to be `fit!` but has not yet been fit.
"""
function glmm(f::Formula, fr::AbstractDataFrame, d::Distribution, l::Link; wt=[], offset=[])
    if d == Binomial() && isempty(wt)
        d = Bernoulli()
    end
    wts = isempty(wt) ? ones(nrow(fr)) : Array(wt)
        # the weights argument is forced to be non-empty in the lmm as it will be used later
    LMM = lmm(f, fr; weights = wts)
    updateL!(setθ!(LMM, getθ(LMM)))
    trms, u, y = LMM.trms, ranef(LMM), copy(model_response(LMM))
    wts = oftype(y, wts)
            # fit a glm to the fixed-effects only
    gl = glm(trms[end - 1], y, d, l; wts = wts, offset = zeros(y))
    r = gl.rr
    β = coef(gl)
    res = GeneralizedLinearMixedModel(LMM, β, copy(β), getθ(LMM), copy.(u), u,
        zeros.(u), gl.rr, similar(y), wts)
    setβθ!(res, vcat(coef(gl), getθ(LMM)))
    LaplaceDeviance!(res)
    res
end

glmm(f::Formula, fr::AbstractDataFrame, d::Distribution) = glmm(f, fr, d, GLM.canonicallink(d))

Base.logdet{T}(m::GeneralizedLinearMixedModel{T}) = logdet(m.LMM)

"""
    LaplaceDeviance{T}(m::GeneralizedLinearMixedModel{T})::T

Return the Laplace approximation to the deviance of `m`.

If the distribution `D` does not have a scale parameter the Laplace approximation
is defined as the squared length of the conditional modes, `u`, plus the determinant
of `Λ'Z'ZΛ + 1`, plus the sum of the squared deviance residuals.
"""
LaplaceDeviance{T}(m::GeneralizedLinearMixedModel{T})::T =
    sum(m.resp.devresid) + logdet(m) + mapreduce(u -> sum(abs2, u), +, m.u)

"""
    LaplaceDeviance!(m::GeneralizedLinearMixedModel)

Update `m.η`, `m.μ`, etc., install the working response and working weights in
`m.LMM`, update `m.LMM.A` and `m.LMM.R`, then evaluate `LaplaceDeviance`.
"""
function LaplaceDeviance!(m::GeneralizedLinearMixedModel)
    updateη!(m)
    GLM.wrkresp!(vec(m.LMM.trms[end]), m.resp)
    reweight!(m.LMM, m.resp.wrkwt)
    LaplaceDeviance(m)
end

function StatsBase.loglikelihood{T}(m::GeneralizedLinearMixedModel{T})
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

lowerbd(m::GeneralizedLinearMixedModel) = vcat(fill(-Inf, size(m.β)), lowerbd(m.LMM))

"""
    updateη!(m::GeneralizedLinearMixedModel)

Update the linear predictor, `m.η`, from the offset and the `B`-scale random effects.
"""
function updateη!(m::GeneralizedLinearMixedModel)
    η, lm, b, u = m.η, m.LMM, m.b,  m.u
    Λ, trms = lm.Λ, lm.trms
    A_mul_B!(η, trms[end - 1], m.β)
    for i in eachindex(b)
        unscaledre!(η, trms[i], A_mul_B!(b[i], Λ[i], u[i]))
    end
    updateμ!(m.resp, η)
    m
end

average{T<:AbstractFloat}(a::T, b::T) = (a + b) / 2

"""
    pirls!(m::GeneralizedLinearMixedModel)

Use Penalized Iteratively Reweighted Least Squares (PIRLS) to determine the conditional
modes of the random effects.

When `varyβ` is true both `u` and `β` are optimized with PIRLS.  Otherwise only `u` is
optimized and `β` is held fixed.

Passing `verbose = true` provides verbose output of the iterations.
"""
function pirls!{T}(m::GeneralizedLinearMixedModel{T}, varyβ::Bool=false, verbose::Bool=false)
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
        varyβ && Ac_ldiv_B!(feL(m), copy!(β, lm.L[end, end - 1]))
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
    setβθ!{T}(m::GeneralizedLinearMixedModel{T}, v::Vector{T})

Set the parameter vector, `:βθ`, of `m` to `v`.

`βθ` is the concatenation of the fixed-effects, `β`, and the covariance parameter, `θ`.
"""
function setβθ!{T}(m::GeneralizedLinearMixedModel{T}, v::Vector{T})
    setβ!(m, v)
    setθ!(m, view(v, (length(m.β) + 1) : length(v)))
end

function setβ!{T}(m::GeneralizedLinearMixedModel{T}, v::Vector{T})
    β = m.β
    copy!(β, view(v, 1 : length(β)))
    m
end

function setθ!{T}(m::GeneralizedLinearMixedModel, v::AbstractVector{T})
    setθ!(m.LMM, copy!(m.θ, v))
    m
end

sdest{T <: AbstractFloat}(m::GeneralizedLinearMixedModel{T}) = convert(T, NaN)

"""
    fit!(m::GeneralizedLinearMixedModel[, verbose = false, fast = false])

Optimize the objective function for `m`.

When `fast` is `true` a potentially much faster but slightly less accurate algorithm, in
which `pirls!` optimizes both the random effects and the fixed-effects parameters,
is used.
"""
function StatsBase.fit!{T}(m::GeneralizedLinearMixedModel{T}; verbose::Bool=false,
    fast::Bool=false)

## FIXME: fast should not be passed as an argument.  Whether or not β is optimized by PIRLS
## should be determined by the length of optsum.initial, lowerbd and final.

    fast || fit!(m, verbose=verbose, fast=true) # use the fast fit first then slow fit to refine

    β = m.β
    lm = m.LMM
    optsum = lm.optsum
    pars = fast ? copy(optsum.initial) : vcat(β, optsum.initial)
    opt = NLopt.Opt(optsum.optimizer, length(pars))

    lb = fast ? optsum.lowerbd : vcat(fill!(similar(β), -Inf), optsum.lowerbd)
    NLopt.lower_bounds!(opt, lb)

    NLopt.ftol_rel!(opt, optsum.ftol_rel) # relative criterion on objective
    NLopt.ftol_abs!(opt, optsum.ftol_abs) # absolute criterion on objective
    NLopt.xtol_rel!(opt, optsum.ftol_rel) # relative criterion on parameter values
#    NLopt.xtol_abs!(opt, optsum.xtol_abs) # absolute criterion on parameter values

    setpar! = fast ? setθ! : setβθ!
    feval = 0
    function obj(x::Vector{T}, g::Vector{T})
        length(g) == 0 || error("gradient not defined for this model")
        feval += 1
        val = pirls!(setpar!(m, x), fast)
        feval == 1 && (optsum.finitial = val)
        verbose && println("f_", feval, ": ", round(val, 5), " ", x)
        val
    end
    NLopt.min_objective!(opt, obj)
    fmin, xmin, ret = NLopt.optimize(opt, pars)
    ## check if very small parameter values bounded below by zero can be set to zero
    xmin_ = copy(xmin)
    for i in eachindex(xmin_)
        if lb[i] == zero(T) && zero(T) < xmin_[i] < T(0.001)
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
    if ret ∈ [:FAILURE, :INVALID_ARGS, :OUT_OF_MEMORY, :ROUNDOFF_LIMITED, :FORCED_STOP]
        warn("NLopt optimization failure: $ret")
    end
    m
end

function Base.show(io::IO, m::GeneralizedLinearMixedModel) # not tested
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

varest{T <: AbstractFloat}(m::GeneralizedLinearMixedModel{T}) = one(T)
