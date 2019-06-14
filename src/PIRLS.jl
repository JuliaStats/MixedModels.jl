using Printf: @sprintf

function StatsBase.dof(m::GeneralizedLinearMixedModel)
    length(m.β) + length(m.θ) + GLM.dispersion_parameter(m.resp.d)
end

StatsBase.fitted(m::GeneralizedLinearMixedModel) = m.resp.mu

function fixef(m::GeneralizedLinearMixedModel{T}, permuted=true) where {T}
    permuted && return m.β
    Xtrm = m.LMM.trms[end - 1]
    iperm = invperm(Xtrm.piv)
    p = length(iperm)
    r = Xtrm.rank
    r == p ? m.β[iperm] : copyto!(fill(-zero(T), p), m.β)[iperm]
end

function GeneralizedLinearMixedModel(f::Formula, fr::AbstractDataFrame,
         d::Distribution, l::GLM.Link; wt=[], offset=[], contrasts = Dict())
    if d == Binomial() && isempty(wt)
        d = Bernoulli()
    end
    LMM = LinearMixedModel(f, fr; weights = wt, contrasts=contrasts, rdist=d)
    X = LMM.trms[end - 1].x
    T = eltype(X)
    y = copy(model_response(LMM))
    if isempty(wt)
        LMM = LinearMixedModel(LMM.formula, LMM.trms, fill!(similar(y), 1), LMM.A, LMM.L, LMM.optsum)
    end
    updateL!(setθ!(LMM, getθ(LMM)))
            # fit a glm to the fixed-effects only - awkward syntax is to by-pass a test
    gl = isempty(wt) ? glm(X, y, d, l) : glm(X, y, d, l, wts=wt)
    β = coef(gl)
    u = [fill(zero(T), vsize(t), nlevs(t)) for t in reterms(LMM)]
    vv = length(u) == 1 ? vec(u[1]) : T[]

    res = GeneralizedLinearMixedModel(LMM, β, copy(β), getθ(LMM), copy.(u), u,
        zero.(u), gl.rr, similar(y), oftype(y, wt), similar(vv),
        similar(vv), similar(vv), similar(vv))
    setβθ!(res, vcat(coef(gl), getθ(LMM)))
    deviance!(res, 1)
    res
end

GeneralizedLinearMixedModel(f::Formula, fr::AbstractDataFrame, d::Distribution;
    wt=[], offset=[], contrasts=Dict()) = GeneralizedLinearMixedModel(f, fr, d,
    GLM.canonicallink(d), wt=wt, offset=offset, contrasts=contrasts)

StatsBase.fit(::Type{GeneralizedLinearMixedModel},
              f::Formula,
              fr::AbstractDataFrame,
              d::Distribution,
              l::GLM.Link = GLM.canonicallink(d);
              wt=[],
              offset=[],
              contrasts=Dict(),
              verbose=false,
              fast=false,
              nAGQ=1) =
    fit!(GeneralizedLinearMixedModel(f, fr, d, l, wt=wt, offset=offset, contrasts=contrasts),
         verbose=verbose, fast=fast, nAGQ=nAGQ)

"""
    deviance(m::GeneralizedLinearMixedModel{T}, nAGQ=1)::T where {T}

Return the deviance of `m` evaluated by adaptive Gauss-Hermite quadrature

If the distribution `D` does not have a scale parameter the Laplace approximation
is defined as the squared length of the conditional modes, `u`, plus the determinant
of `Λ'Z'WZΛ + I`, plus the sum of the squared deviance residuals.
"""
function StatsBase.deviance(m::GeneralizedLinearMixedModel{T}, nAGQ=1) where {T}
    nAGQ == 1 && return T(sum(m.resp.devresid) + logdet(m) + sum(u -> sum(abs2, u), m.u))
    u = vec(m.u[1])
    u₀ = vec(m.u₀[1])
    copyto!(u₀, u)
    ra = RaggedArray(m.resp.devresid, m.LMM.trms[1].refs)
    devc0 = sum!(map!(abs2, m.devc0, u), ra)  # the deviance components at z = 0
    sd = map!(inv, m.sd, m.LMM.L.data[Block(1,1)].diag)
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
    GLM.wrkresp!(vec(m.LMM.trms[end].x), m.resp)
    reweight!(m.LMM, m.resp.wrkwt)
    deviance(m, nAGQ)
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

function lowerbd(m::GeneralizedLinearMixedModel)
    lb = lowerbd(m.LMM)
    vcat(fill(eltype(lb)(-Inf), size(m.β)), lb)
end

StatsBase.nobs(m::GeneralizedLinearMixedModel) = length(m.η)

StatsBase.predict(m::GeneralizedLinearMixedModel) = fitted(m)

updatedevresid!(r::GLM.GlmResp, η::AbstractVector) = GLM.updateμ!(r, η)

fastlogitdevres(η, y) = 2log1p(exp(iszero(y) ? η : -η))

function updatedevresid!(r::GLM.GlmResp{V,<:Bernoulli,LogitLink}, η::V) where V<:AbstractVector{<:AbstractFloat}
    map!(fastlogitdevres, r.devresid, η, r.y)
    r
end

"""
    updateη!(m::GeneralizedLinearMixedModel)

Update the linear predictor, `m.η`, from the offset and the `B`-scale random effects.
"""
function updateη!(m::GeneralizedLinearMixedModel)
    η, b, u = m.η, m.b, m.u
    trms = m.LMM.trms
    mul!(η, trms[end - 1].x, m.β)
    for i in eachindex(b)
        unscaledre!(η, trms[i], mul!(b[i], trms[i].Λ, u[i]))
    end
    GLM.updateμ!(m.resp, η)
    m
end

average(a::T, b::T) where {T<:AbstractFloat} = (a + b) / 2

"""
    pirls!(m::GeneralizedLinearMixedModel)

Use Penalized Iteratively Reweighted Least Squares (PIRLS) to determine the conditional
modes of the random effects.

When `varyβ` is true both `u` and `β` are optimized with PIRLS.  Otherwise only `u` is
optimized and `β` is held fixed.

Passing `verbose = true` provides verbose output of the iterations.
"""
function pirls!(m::GeneralizedLinearMixedModel{T}, varyβ=false, verbose=false) where {T}
    iter, maxiter, obj = 0, 100, T(-Inf)
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
            print(", β =")
            show(β)
        end
        println()
    end

    while iter < maxiter
        iter += 1
        varyβ && ldiv!(adjoint(feL(m)), copyto!(β, lm.L.data.blocks[end, end - 1]))
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

sdest(m::GeneralizedLinearMixedModel{T}) where {T} = convert(T, NaN)

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
            println("f", lpad(feval,5), ": ", val, ", ", x)
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

function Base.show(io::IO, m::GeneralizedLinearMixedModel)
    nAGQ = m.LMM.optsum.nAGQ
    println(io, "Generalized Linear Mixed Model fit by maximum likelihood (nAGQ = $nAGQ)")
    println(io, "  ", m.LMM.formula)
    println(io, "  Distribution: ", Distribution(m.resp))
    println(io, "  Link: ", GLM.Link(m.resp), "\n")
    println(io, string("  Deviance: ", @sprintf("%.4f", deviance(m, nAGQ))), "\n")

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

varest(m::GeneralizedLinearMixedModel{T}) where {T} = one(T)
