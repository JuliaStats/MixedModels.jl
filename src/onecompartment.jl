using LinearAlgebra, Tables

"""
    onecompartment!(grad, dose, time, K, Ka, Cl)

Return the predicted concentration for a one-compartment model
given scalar `dose`, `time`, and the model parameters `K`, `Ka`, `Cl`.
Overwrite `grad` with the gradient.

This function is based on the output of the R function call
    deriv(~ dose*K*Ka*(exp(-K*time)-exp(-Ka*time))/(Cl*(Ka - K)), c("K","Ka","Cl"))
which performs symbolic differentiation followed by common subexpression elimination.
"""
function onecompartment(dose::T, t::T, K::T, Ka::T, Cl::T) where T<:AbstractFloat
    _expr1 = dose * K
    _expr2 = _expr1 * Ka
    _expr5 = exp(-K * t)
    _expr8 = exp(-Ka * t)
    _expr9 = _expr5 - _expr8
    _expr10 = _expr2 * _expr9
    _expr11 = Ka - K
    _expr12 = Cl * _expr11
    _expr21 = _expr12^2
    _expr22 = _expr10 * Cl/_expr21
    _expr10/_expr12,
    (K = (dose * Ka * _expr9 - _expr2 * (_expr5 * t))/_expr12 + _expr22,
     Ka = (_expr1 * _expr9 + _expr2 * (_expr8 * t))/_expr12 - _expr22,
     Cl = -(_expr10 * _expr11/_expr21))
end

function onecompartment(dose::T, t::T,
        logpars::AbstractVector{T}) where {T<:AbstractFloat}
    K, Ka, Cl = exp.(logpars)
    μ, g = onecompartment(dose, t, K, Ka, Cl)
    μ, (lK = g.K * K, lKa = g.Ka * Ka, lCl = g.Cl * Cl)
end

"""
    resgrad!(μ, resid, Jac, df, β)

Update the mean response, `μ`, the residual, `resid` and the transpose of
the Jacobian, `Jac`, given `df`, an object like a `DataFrame` for which `Tables.rows`
returns a NamedTuple including names `dose`, `time`, and `conc`.

Returns the sum of squared residuals.
""" 
function resgrad!(μ, resid, Jac, df, β)
    rss = zero(eltype(μ))
    for (i,r) in enumerate(Tables.rows(df))
        μ[i], g = onecompartment(r.dose, r.time, β)
        resi = resid[i] = r.conc - μ[i]
        rss += abs2(resi)
        Jac[i,1] = g.lK
        Jac[i,2] = g.lKa
        Jac[i,3] = g.lCl
    end
    rss
end

function resgradre!(μ, resid, Jac, df, β, b)
    ϕ = similar(β)
    grad = similar(β)
    oldsubj = zero(eltype(df.subj))
    for (i,r) in enumerate(Tables.rows(df))
        if r.subj ≠ oldsubj
            oldsubj = r.subj
            for j in eachindex(β)
                ϕ[j] = β[j] + b[j, oldsubj]
            end
        end
        μ[i] = onecompartment!(grad, r.dose, r.time, ϕ)
        resid[i] = r.conc - μ[i]
        for j in eachindex(grad)
            Jac[i,j] = grad[j]
        end
    end
    resid
end

## Simple nonlinear least squares
function increment!(δ, resid, Jac)
    m, n = size(Jac)
    fac = qr!(Jac)
    lmul!(fac.Q', resid)
    for i in eachindex(δ)
        δ[i] = resid[i]
    end
    ldiv!(UpperTriangular(fac.R), δ)
    sum(abs2, view(resid, 1:n)) / sum(abs2, view(resid, (n+1):m))
end

function nls!(β, df)
    δ = similar(β)     # parameter increment
    b = copy(β)        # trial parameter value
    n = size(df, 1)
    μ = similar(β, n)
    resid = similar(μ)
    Jac = similar(β, (n, 3))
    oldrss = resgrad!(μ, resid, Jac, df, β)
    cvg = increment!(δ, resid, Jac)
    tol = 0.0001       # convergence criterion tolerance
    minstep = 0.001    # minimum step factor
    maxiter = 100      # maximum number of iterations
    iter = 1
    while cvg > tol && iter ≤ maxiter
        step = 1.0     # step factor
        b .= β .+ step .* δ
        rss = resgrad!(μ, resid, Jac, df, b)
        while rss > oldrss && step ≥ minstep  # step-halving to ensure reduction of rss
            step /= 2
            b .= β .+ step .* δ
            rss = resgrad!(μ, resid, Jac, df, b)
        end
        if step < minstep
            throw(ErrorException("Step factor reduced below minstep of $minstep"))
        end
        copy!(β, b)
        cvg = increment!(δ, resid, Jac)
        iter += 1
        oldrss = rss
    end
    if iter > maxiter
        throw(ErrorException("Maximum number of iterations, $maxiter, exceeded"))
    end
    (lK = b[1], lKa = b[2], lCl = b[3])
end

function updateTerms!(m::LinearMixedModel, resid, Jac)
    copyto!(first(m.feterms).x, Jac)
    copyto!(last(m.feterms).x, resid)
    re = first(m.reterms)
    transpose!(re.z, Jac)
    copyto!(re.adjA.nzval, re.z)
    terms = vcat(m.reterms, m.feterms)
    k = length(terms)
    A = m.A
    for j in 1:k
        for i in j:k
            mul!(A[Block(i,j)], terms[i]', terms[j])
        end
    end
    updateL!(m)
    L = m.L
    prss = abs2(first(L[Block(3,3)]))
    prss, (sum(abs2, L[Block(3,1)]) + sum(abs2, L[Block(3,2)])) / prss
end

function pnls!(m::LinearMixedModel, β, b, df)
    δ = fill!(similar(β), 0)     # parameter increment
    β₀ = copy(β)       # trial parameter value
    b₀ = copy(b)
    δb = [fill!(similar(b), 0)]
    n = size(df, 1)
    μ = similar(β, n)
    resid = similar(μ)
    Jac = similar(β, (n, 3))
    resgradre!(μ, resid, Jac, df, β, b)
    oldprss, cvg = updateTerms!(m, resid, Jac)
    fixef!(δ, m)
    ranef!(δb, m, δ, false)
    tol = 0.0001       # convergence criterion tolerance
    minstep = 0.001    # minimum step factor
    maxiter = 100      # maximum number of iterations
    iter = 1
    while cvg > tol && iter ≤ maxiter
        step = 1.0                              # step factor
        β .= β₀ .+ step .* δ
        b .= b₀ .+ step .* first(δb)
        resgradre!(μ, resid, Jac, df, β, b)
        prss, cvg = updateTerms!(m, resid, Jac)
        while prss > oldprss && step ≥ minstep  # step-halving to ensure reduction of rss
            step /= 2
            β .= β₀ .+ step .* δ
            b .= b₀ .+ step .* first(δb)
            resgradre!(μ, resid, Jac, df, β, b)
            prss, cvg = updateTerms!(m, resid, Jac)
        end
        if step < minstep
            throw(ErrorException("Step factor reduced below minstep of $minstep"))
        end
        copyto!(β₀, β)
        copyto!(b₀, b)
        fixef!(δ, m)
        ranef!(δb, m, δ, false)
        iter += 1
        oldprss = prss
    end
    if iter > maxiter
        throw(ErrorException("Maximum number of iterations, $maxiter, exceeded"))
    end
    objective(lmm)
end