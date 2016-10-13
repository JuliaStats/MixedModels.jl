"""
    GeneralizedLinearMixedModel

Generalized linear mixed-effects model representation

Members:

- `LMM`: a [`LinearMixedModel`](@ref) - used for the random effects only.
- `β`: the fixed-effects vector
- `θ`: covariance parameter vector
- `b`: similar to `u`, equivalent to `broadcast!(*, b, LMM.Λ, u)`
- `u`: a vector of matrices of random effects
- `u₀`: similar to `u`.  Used in the PIRLS algorithm if step-halving is necessary.
- `X`:
- `resp`: a `GlmResp` object
- `η`: the linear predictor
- `offset₀`: prior offset; `T[]` is allowed
- `wt`: vector of prior case weights, a value of `T[]` indicates equal weights.
"""

immutable GeneralizedLinearMixedModel{T <: AbstractFloat} <: MixedModel
    LMM::LinearMixedModel{T}
    β::Vector{T}
    θ::Vector{T}
    b::Vector{Matrix{T}}
    u::Vector{Matrix{T}}
    u₀::Vector{Matrix{T}}
    X::Matrix{T}
    resp::GlmResp
    η::Vector{T}
    offset₀::Vector{T}
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
    A, R, trms, u, y = LMM.A, LMM.R, LMM.trms, ranef(LMM), copy(model_response(LMM))
    wts = oftype(y, wts)
    kp1 = length(LMM.Λ) + 1
    X = trms[kp1]
            # zero the dimension of the fixed-effects in trms, A and R
    trms[kp1] = zeros(length(y), 0)
    LMM.wttrms[kp1] = trms[kp1]
    for i in 1:kp1
        qi = size(trms[i], 2)
        A[i, kp1] = zeros((qi, 0))
        R[i, kp1] = zeros((qi, 0))
    end
    qend = size(trms[end], 2)  # should always be 1 but no harm in extracting it
    A[kp1, end] = zeros((0, qend))
    R[kp1, end] = zeros((0, qend))
            # fit a glm to the fixed-effects only
    gl = glm(X, y, d, l; wts = wts, offset = zeros(y))
    r = gl.rr
    res = GeneralizedLinearMixedModel(LMM, coef(gl), getθ(LMM), copy.(u), u,
        zeros.(u), X, gl.rr, similar(y), oftype(y, offset), wts)
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
    wrkresp(vec(m.LMM.trms[end]), m.resp)
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

function restoreX!(m::GeneralizedLinearMixedModel)
    if !isempty(m.LMM.R[end - 1, end - 1])
        return m
    end
    lm, X = m.LMM, m.X
    A, R, trms, k = lm.A, lm.R, lm.trms, length(lm.Λ)
    kp1 = k + 1
    trms[kp1] = X
    lm.wttrms[kp1] = copy(X)
    for i in 1 : kp1
        A[i, kp1] = trms[i]'X
        R[i, kp1] = copy(A[i, kp1])
    end
    A[kp1, end] = X'trms[end]
    R[kp1, end] = copy(A[kp1, end])
    reweight!(lm, m.resp.wrkwt)
end

"""
    updateη!(m::GeneralizedLinearMixedModel)

Update the linear predictor, `m.η`, from the offset and the `B`-scale random effects.
"""
function updateη!(m::GeneralizedLinearMixedModel)
    η, lm, b, u = m.η, m.LMM, m.b,  m.u
    Λ, trms = lm.Λ, lm.trms
    isempty(m.offset₀) ? fill!(η, 0) : copy!(η, m.offset₀)
    for i in eachindex(b)
        unscaledre!(η, trms[i], A_mul_B!(Λ[i], copy!(b[i], u[i])))
    end
    updateμ!(m.resp, η)
    m
end

"""
    pirls!(m::GeneralizedLinearMixedModel)

Use Penalized Iteratively Reweighted Least Squares (PIRLS) to determine the conditional
modes of the random effects.
"""
function pirls!{T}(m::GeneralizedLinearMixedModel{T})
    iter, maxiter, obj = 0, 100, T(-Inf)
    u₀, u = m.u₀, m.u
    for j in eachindex(u)         # start from u all zeros
        copy!(u₀[j], fill!(u[j], 0))
    end
    obj₀ = LaplaceDeviance!(m) * 1.0001
    while iter < maxiter
        iter += 1
        ranef!(u, m.LMM, true)    # solve for new values of u
        obj = LaplaceDeviance!(m) # update GLM vecs and evaluate Laplace approx
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
                ui = u[i]
                ui₀ = u₀[i]
                for j in eachindex(ui)
                    ui[j] += ui₀[j]
                    ui[j] *= 0.5
                end
            end
            obj = LaplaceDeviance!(m)
        end
        if isapprox(obj, obj₀; atol = 0.0001)
            break
        end
        for i in eachindex(u)
            copy!(u₀[i], u[i])
        end
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
    β, lm, offset₀, X, rr = m.β, m.LMM, m.offset₀, m.X, m.resp
    lb = length(β)
    copy!(β, view(v, 1 : lb))
    setθ!(m.LMM, copy!(m.θ, view(v, (lb + 1) : length(v))))
    BLAS.gemv!('N', one(T), X, β, one(T), isempty(offset₀) ? fill!(rr.offset, 0) : copy!(rr.offset, offset₀))
    m
end

sdest{T <: AbstractFloat}(m::GeneralizedLinearMixedModel{T}) = one(T)

"""
    fit!(m::GeneralizedLinearMixedModel[, verbose = false, optimizer=:LN_BOBYQA]])

Optimize the objective function for `m`
"""
function StatsBase.fit!(m::GeneralizedLinearMixedModel, verbose::Bool=false, optimizer::Symbol=:LN_BOBYQA)
    β, lm = m.β, m.LMM
    βθ = vcat(β, getθ(lm))
#    @show(βθ)
    opt = NLopt.Opt(optimizer, length(βθ))
    NLopt.ftol_rel!(opt, 1e-12)   # relative criterion on deviance
    NLopt.ftol_abs!(opt, 1e-8)    # absolute criterion on deviance
    NLopt.xtol_abs!(opt, 1e-10)   # criterion on parameter value changes
    NLopt.lower_bounds!(opt, vcat(fill!(similar(β), -Inf), lowerbd(lm)))
    feval = 0
    function obj(x::Vector{Float64}, g::Vector{Float64})
        if length(g) ≠ 0
            error("gradient not defined for this model")
        end
        feval += 1
        setβθ!(m, x) |> pirls!
    end
    if verbose
        function vobj(x::Vector{Float64}, g::Vector{Float64})
            if length(g) ≠ 0
                error("gradient not defined for this model")
            end
            feval += 1
            val = setβθ!(m, x) |> pirls!
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
    fmin, xmin, ret = NLopt.optimize(opt, βθ)
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
    m.LMM.opt = OptSummary(βθ,xmin,fmin,feval,optimizer)
    restoreX!(m)
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
