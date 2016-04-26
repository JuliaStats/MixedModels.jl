## Penalized iteratively reweighted least squares algorithm for determining the
## conditional modes of the random effects in a GLMM

type GeneralizedLinearMixedModel{T <: AbstractFloat, D <: UnivariateDistribution, L <: Link} <: MixedModel
    LMM::LinearMixedModel{T}
    dist::D
    link::L
    β₀::DenseVector{T}
    u::Vector
    u₀::Vector
    X::DenseMatrix{T}
    y::DenseVector{T}
    μ::DenseVector{T}
    η::DenseVector{T}
    dμdη::DenseVector{T}
    devresid::DenseVector{T}
    offset::DenseVector{T}
    var::DenseVector{T}
    wrkresid::DenseVector{T}
    wrkwt::DenseVector{T}
    wt::DenseVector{T}
    devold::T
end

function glmm(f::Formula, fr::AbstractDataFrame, d::Distribution, wt::Vector, l::Link)
    if d == Binomial() && isempty(wt)
        d = Bernoulli()
    end
    wts = isempty(wt) ? ones(nrow(fr)) : wt
    LMM = lmm(f, fr; weights = wts)
    A, R, trms, u, y = LMM.A, LMM.R, LMM.trms, ranef(LMM, true), copy(model_response(LMM))
    kp1 = length(LMM.Λ) + 1
    X = copy(trms[kp1])         # the copy may be unnecessary
            # zero the dimension of the fixed-effects in trms, A and R
    trms[kp1] = zeros((length(y), 0))
    for i in 1:kp1
        qi = size(trms[i], 2)
        A[i, kp1] = zeros((qi, 0))
        R[i, kp1] = zeros((qi, 0))
    end
    qend = size(trms[end], 2)  # should always be 1 but no harm in extracting it
    A[kp1, end] = zeros((0, qend))
    R[kp1, end] = zeros((0, qend))
            # fit a glm pm the fixed-effects only
    gl = glm(X, y, d, l; wts = wts)
    r = gl.rr
    β₀ = coef(gl)
    res = GeneralizedLinearMixedModel(LMM, d, l, β₀, u, map(zeros, u), X, y, r.mu,
        r.eta, r.mueta, r.devresid, X * β₀, r.var, r.wrkresid, r.wrkwts, r.wts, zero(eltype(X)))
    updateμ!(res)
    wrkresp!(trms[end], res)
    sqrtwts = LMM.sqrtwts = map(sqrt, res.wrkwt)
    trms, wttrms = LMM.trms, LMM.wttrms
    for i in eachindex(trms)
        wttrms[i] = scale(sqrtwts, trms[i])
    end
    reweight!(LMM, res.wrkwt)
    fit!(LMM)
    res.devold = deviance(gl) + logdet(LMM)
    ranef!(res.u, LMM, true)
    LaplaceDeviance!(res)
    res
end

function glmm(f::Formula, fr::AbstractDataFrame, d::Distribution, wt::DataVector, l::Link)
    glmm(f, fr, d, convert(Vector, wt), l)
end

glmm(f::Formula, fr::AbstractDataFrame, d::Distribution, wt::Vector) = glmm(f, fr, d, wt, GLM.canonicallink(d))

glmm(f::Formula, fr::AbstractDataFrame, d::Distribution) = glmm(f, fr, d, Float64[])

lmm(m::GeneralizedLinearMixedModel) = m.LMM

Base.logdet(m::GeneralizedLinearMixedModel) = logdet(lmm(m))

"""
    LaplaceDeviance(m)

Laplace approximation to the deviance of a GLMM

Args:

- `m`: a `GeneralizedLinearMixedModel`

Returns:
  the Laplace approximation to negative twice the log-likelihood of `m`
"""
function LaplaceDeviance(m::GeneralizedLinearMixedModel)
    dd, μ, y = typeof(m.dist), m.μ, m.y
    s = mapreduce(sumabs2, +, m.u) + logdet(m)
    if dd ≠ Binomial
        for i in eachindex(y)
            s -= 2 * logpdf(dd(μ[i]), y[i])
        end
        return s
    end
    n = m.wt
    for i in eachindex(n)
        s -= 2 * logpdf(dd(n[i], μ[i]), round(Int, y[i] * n[i]))
    end
    s
end

function LaplaceDeviance!(m::GeneralizedLinearMixedModel)
    updateη!(m)
    lm = lmm(m)
    wrkresp!(lm.trms[end], m)
    reweight!(lm, m.wrkwt)
    lm[:θ] = lm[:θ]  # FIXME: this is for side-effect of updating lm.R.  Make it explicit
    LaplaceDeviance(m)
end

function updateη!(m::GeneralizedLinearMixedModel)
    lm = lmm(m)
    η, u, Λ, trms = m.η, m.u, lm.Λ, lm.trms
    if size(trms[end - 1], 2) != 0
        throw(ArgumentError("fixed-effects model matrix in lmm(m) should have 0 columns"))
    end
    fill!(η, 0)
    for i in eachindex(u)
        unscaledre!(η, trms[i], Λ[i], u[i])
    end
    updateμ!(m)
end

"""
    pirls!(m)

Use Penalized Iteratively Reweighted Least Squares (PIRLS) to determine the conditional modes of the random effects

Args:

- `m`: a `GeneralizedLinearMixedModel`

Returns:
  the updated model `m`

Note:
  On entry the values of `m.u₀` and `m.devold` should correspond.
  One safe approach is to zero out `m.u₀` and evaluate devold from fixed-effects only.
"""
function pirls!(m::GeneralizedLinearMixedModel)
    u₀, u, obj₀, obj, lm, iter, maxiter = m.u₀, m.u, m.devold, m.devold, lmm(m), 0, 100
    while iter < maxiter
        iter += 1
        nhalf = 0
        obj = LaplaceDeviance!(m)
        while obj >= obj₀
            nhalf += 1
            if nhalf > 10
                throw(ErrorException("number of averaging steps > 10"))
            end
            for i in eachindex(u)
                ui = u[i]
                ui₀ = u₀[i]
                for j in eachindex(ui)
                    ui[j] += ui₀[j]
                    ui[j] *= 0.5
                end
            end
            obj = LaplaceDeviance!(m)
        end
        for i in eachindex(u)
            copy!(u₀[i], u[i])
        end
        ranef!(u, lm, true)
        if isapprox(obj, obj₀; rtol = 0.00001, atol = 0.0001)
            break
        end
        obj₀ = obj
    end
    obj
end

function Base.setindex!{T <: AbstractFloat}(m::GeneralizedLinearMixedModel, v::Vector{T}, k::Symbol)
    if k ≠ :βθ
        throw(ArgumentError(":βθ is the only key allowed for a GeneralizedLinearMixedModel"))
    end
    β, lm, u, u₀ = m.β₀, lmm(m), m.u, m.u₀
    lb = length(β)
    copy!(β, v[1:lb])
    lm[:θ] = v[(lb + 1):length(v)]
    A_mul_B!(m.offset, m.X, β)
    for i in eachindex(u₀)
        copy!(u[i], fill!(u₀[i], zero(T)))
    end
    m.devold = LaplaceDeviance!(m)
    ranef!(m.u, lm, true)
end

"""
    fit!(m[, verbose = false])

Optimize the objective of a `GeneralizedLinearMixedModel` using an NLopt optimizer.

Args:
- `m`: a [`GeneralizedLinearMixedModel`]({ref})
- `verbose`: `Bool` indicating if information on iterations should be printed, Defaults to `false`

Named Args:
- `optimizer`: `Symbol` form of the name of a derivative-free optimizer in `NLopt` that allows for
  box constraints.  Defaults to `:LN_BOBYQA`
"""
function StatsBase.fit!(m::GeneralizedLinearMixedModel, verbose::Bool=false, optimizer::Symbol=:LN_BOBYQA)
    β, lm = m.β₀, lmm(m)
    βΘ = vcat(β, lm[:θ])
    opt = NLopt.Opt(optimizer, length(βΘ))
    NLopt.ftol_rel!(opt, 1e-12)   # relative criterion on deviance
    NLopt.ftol_abs!(opt, 1e-8)    # absolute criterion on deviance
    NLopt.xtol_abs!(opt, 1e-10)   # criterion on parameter value changes
    NLopt.lower_bounds!(opt, vcat(-Inf * ones(β), lowerbd(lm)))
    feval = 0
    function obj(x::Vector{Float64}, g::Vector{Float64})
        if length(g) ≠ 0
            error("gradient not defined for this model")
        end
        feval += 1
        m[:βθ] = x
        pirls!(m)
    end
    if verbose
        function vobj(x::Vector{Float64}, g::Vector{Float64})
            if length(g) ≠ 0
                error("gradient not defined for this model")
            end
            feval += 1
            m[:βθ] = x
            val = pirls!(m)
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
    fmin, xmin, ret = NLopt.optimize(opt, βΘ)
    ## very small parameter values often should be set to zero
#    xmin1 = copy(xmin)
#    modified = false
#    for i in eachindex(xmin1)
#        if 0. < abs(xmin1[i]) < 1.e-5
#            modified = true
#            xmin1[i] = 0.
#        end
#    end
#    if modified  # branch not tested
#        m[:θ] = xmin1
#        ff = objective(m)
#        if ff ≤ (fmin + 1.e-5)  # zero components if increase in objective is negligible
#            fmin = ff
#            copy!(xmin,xmin1)
#        else
#            m[:θ] = xmin
#        end
#    end
#    m.opt = OptSummary(th,xmin,fmin,feval,optimizer)
    if verbose
        println(ret)
    end
    m
end
