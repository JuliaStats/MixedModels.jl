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

immutable GeneralizedLinearMixedModel{T <: AbstractFloat} <: MixedModel
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
    setθ!(LMM, getθ(LMM)) |> cfactor!
    A, R, trms, u, y = LMM.A, LMM.R, LMM.trms, ranef(LMM), copy(model_response(LMM))
    wts = oftype(y, wts)
            # fit a glm to the fixed-effects only
    gl = glm(LMM.trms[end - 1], y, d, l; wts = wts, offset = zeros(y))
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
    sum(m.resp.devresid) + logdet(m) + mapreduce(sumabs2, +, m.u)

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
    accum - (mapreduce(sumabs2, + , m.u) + logdet(m)) / 2
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
    u₀, u, β, β₀, lm = m.u₀, m.u, m.β, m.β₀, m.LMM
    for j in eachindex(u)         # start from u all zeros
        copy!(u₀[j], fill!(u[j], 0))
    end
    varyβ && copy!(β₀, β)
    obj₀ = LaplaceDeviance!(m) * 1.0001
    verbose && @show(varyβ, obj₀, β)

    while iter < maxiter
        iter += 1
        varyβ && A_ldiv_B!(feR(m), copy!(β, lm.R[end - 1, end]))
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
        if isapprox(obj, obj₀; atol = 0.0001)
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

sdest{T <: AbstractFloat}(m::GeneralizedLinearMixedModel{T}) = one(T)

"""
    fit!(m::GeneralizedLinearMixedModel[, verbose = false, optimizer=:LN_BOBYQA]])

Optimize the objective function for `m`
"""
function StatsBase.fit!{T}(m::GeneralizedLinearMixedModel{T}; verbose::Bool=false,
    nAGQ::Integer=1, optimizer::Symbol=:LN_BOBYQA)
    if nAGQ > 0
        fit!(m; verbose=verbose, nAGQ=0, optimizer=optimizer)
    end
    β, lm = m.β, m.LMM
    pars = nAGQ == 0 ? getθ(lm) : vcat(β, getθ(lm))
    lb = lowerbd(nAGQ == 0 ? lm : m)
    opt = NLopt.Opt(optimizer, length(pars))
    NLopt.ftol_rel!(opt, 1e-12)   # relative criterion on deviance
    NLopt.ftol_abs!(opt, 1e-8)    # absolute criterion on deviance
    NLopt.xtol_abs!(opt, 1e-10)   # criterion on parameter value changes
    NLopt.lower_bounds!(opt, lb)
    feval = 0
    function obj(x::Vector{T}, g::Vector{T})
        length(g) == 0 || error("gradient not defined for this model")
        feval += 1
        nAGQ == 0 ? pirls!(setθ!(m, x), true) : pirls!(setβθ!(m, x))
    end
    if verbose
        function vobj(x::Vector{T}, g::Vector{T})
            length(g) == 0 || error("gradient not defined for this model")
            feval += 1
            val = nAGQ == 0 ? pirls!(setθ!(m, x), true, true) : pirls!(setβθ!(m, x))
            print("f_$feval: $(round(val,5)), [")
            showcompact(x[1])
            for i in 2:length(x) print(","); showcompact(x[i]) end
            println("]")
            val
        end
        NLopt.min_objective!(opt, vobj)
    else
        NLopt.min_objective!(opt, obj)
    end
    fmin, xmin, ret = NLopt.optimize(opt, pars)
    ## check if very small parameter values bounded below by zero can be set to zero
    xmin1, fev, modified = copy(xmin), feval, false
    for i in eachindex(xmin1)
        if lb[i] == 0 && 0 < xmin1[i] < 1.e-4
            modified = true
            xmin1[i] = 0.
        end
    end
    if modified
        if (zeroobj = obj(xmin1, T[])) ≤ (fmin + 1.e-5)
            fmin = zeroobj
            copy!(xmin, xmin1)
        else
            obj(xmin, T[])
        end
    end
    m.LMM.opt = OptSummary(pars, xmin, fmin, fev, optimizer)
    if verbose
        println(ret)
    end
    m
end

function VarCorr(m::GeneralizedLinearMixedModel)
    Λ, trms = m.LMM.Λ, m.LMM.trms
    VarCorr(Λ, [string(trms[i].fnm) for i in eachindex(Λ)],
        [trms[i].cnms for i in eachindex(Λ)], NaN)
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
