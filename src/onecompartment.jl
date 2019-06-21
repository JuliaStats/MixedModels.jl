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
function onecompartment!(grad::AbstractVector{T}, dose::T, time::T,
        K::T, Ka::T, Cl::T) where T<:AbstractFloat
    _expr1 = dose * K
    _expr2 = _expr1 * Ka
    _expr5 = exp(-K * time)
    _expr8 = exp(-Ka * time)
    _expr9 = _expr5 - _expr8
    _expr10 = _expr2 * _expr9
    _expr11 = Ka - K
    _expr12 = Cl * _expr11
    _expr21 = _expr12^2
    _expr22 = _expr10 * Cl/_expr21
    grad[1] = (dose * Ka * _expr9 - _expr2 * (_expr5 * time))/_expr12 + _expr22
    grad[2] = (_expr1 * _expr9 + _expr2 * (_expr8 * time))/_expr12 - _expr22
    grad[3] = -(_expr10 * _expr11/_expr21)
    _expr10/_expr12
end

function onecompartment!(grad::AbstractVector{T}, dose::T, time::T,
        logpars::AbstractVector{T}) where {T<:AbstractFloat}
    K, Ka, Cl = pars = exp.(logpars)
    μ = onecompartment!(grad, dose, time, K, Ka, Cl)
    grad .*= pars
    μ
end

"""
    resgrad!(μ, resid, Jt, df, β)

Update the mean response, `μ`, the residual, `resid` and the transpose of
the Jacobian, `Jac`, given `df`, an object like a `DataFrame` for which `Tables.rows`
returns a NamedTuple including names `dose`, `time`, and `conc`.

Returns the sum of squared residuals.
""" 
function resgrad!(μ, resid, Jac, df, β)
    grad = similar(μ, 3)
    rss = zero(eltype(grad))
    for (i,r) in enumerate(Tables.rows(df))
        μ[i] = onecompartment!(grad, r.dose, r.time, β)
        resi = resid[i] = r.conc - μ[i]
        rss += abs2(resi)
        for j in 1:3
            Jac[i,j] = grad[j]
        end
    end
    rss
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
