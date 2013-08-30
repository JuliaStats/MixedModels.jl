using Base.LinAlg.BLAS: trsv!
using Base.LinAlg.LAPACK: gemqrt!,geqrt3!, potri!
using DataFrames, RDatasets, NumericExtensions
import Distributions.fit, Base.show
import NumericExtensions: evaluate, result_type

#const conc = [2.84, 6.57, 10.5, 9.66, 8.58, 8.36, 7.47, 6.89, 5.94, 3.28]

abstract NonlinearRegModel              # nonlinear regression model

abstract NLRegJac <: NonlinearRegModel  # nonlinear regression model with Jacobian

abstract NLRegFD <: NonlinearRegModel   # finite-difference nonlinear regression model

function elCol(x::Array,i::Integer)
    isa(x,Vector) ? x[i] : (isa(x,Matrix) ? x[i,:] : error("x not Vector or Matrix"))
end

abstract Ptransform

type VClka2Vkka <: Ptransform end
function pars!{T<:FloatingPoint}(t::VClka2Vkka,out::Matrix{T},in::Matrix{T})
    copy!(out, in)
    n,p = size(out); p == 3 || error("3 columns required in 'in' and 'out'")
    for i in 1:n
        @inbounds out[i,2] = in[i,2]/in[i,1]
    end
    out
end
pars{T<:FloatingPoint}(t::VClka2Vkka,in::Matrix{T})=pars!(t,similar(in),in)
function parsjac!{T<:FloatingPoint}(::VClka2Vkka,out::Matrix{T},jac::Array{T,3},in::Matrix{T})
    copy!(out, in); n,p = size(out)
    p == 3 || error("3 columns required in 'in' and 'out'")
    (n,p,p) == size(jac) || error("dimension mismatch")
    for i in 1:n
        V = in[i,1]
        Cl = in[i,2]
        out[i,2] = Cl/V
        jac[i,2,1] = -Cl/(V * V)
        jac[i,2,2] = 1. / V
    end
    out,jac
end
pnmsin(::Type{VClka2Vkka}) = ["V","Cl","ka"]
pnmsout(::Type{VClka2Vkka}) = ["V","k","ka"]
    

# Oral, single dose, 1 compartment using k and ka as parameters.
type OralSd1Vkka{T<:FloatingPoint} <: NLRegFD
    time::Vector{Float64}
    dose::Float64
end
nobs(m::OralSd1Vkka) = length(m.time)
pnames(m::OralSd1Vkka) = ["V","k","ka"]
npar(m::OralSd1Vkka) = 3

# Evaluation functor.  Result should be multiplied by dose/V
type fOralSd1kka <: TernaryFunctor end
function evaluate{T<:FloatingPoint}(::fOralSd1kka,t::T,k::T,ka::T)
    ka*(exp(-k*t)-exp(-ka*t))/(ka-k)
end
result_type{T<:FloatingPoint}(::fOralSd1kka,::Type{T},::Type{T},::Type{T}) = T


function expctd!{T<:FloatingPoint}(mu::Vector{T},m::OralSd1Vkka,x::Array{T})
    map1!(Multiply(), map!(fOralSd1Vkka(), mu, m.time, elcol(x,2), elcol(x,3), m.dose ./ elcol(x,1)))
end
# PK model for single oral dose, 1 compartment with parameters V, Cl and ka
type OralSd1VClka <: NLRegFD
    time::Vector{Float64}
    dose::Float64
end
OralSd1VClka(time::Vector{Float64}) = OralSd1VClka(time,1.) # default is unit dose
nobs(m::OralSd1VClka) = length(m.time)
pnames(m::OralSd1VClka) = ["V","Cl","ka"]
npar(m::OralSd1VClka) = 3

function expctd!{T<:Float64}(mu::Vector{T}, m::OralSd1VClka, x::Vector{T})
    length(x) == 3 || error("trailing dimension of x must be 3")
    map1!(Multiply(), map!(fOralSd1kka(), mu, m.time, x[2]/x[1], x[3]), m.dose ./ x[1])
end

function expctd!{T<:Float64}(eta::Vector{T}, m::OralSd1VClka, x::Matrix{T})
    n,p = size(x); p == 3 || error("trailing dimension of x must be 3")
#    V = x[1,:]; Cl = x[2,:]; ka = x[3,:]; k = Cl./V;
    map1!(Multiply(), map!(fOralSd1kka(),m.time,x[2,:]./x[1,:],x[3,:]), m.dose./x[1,:])
end
expctd{T<:Float64}(m::NonlinearRegModel,x::Vector{T}) = expctd!(Array(T,(nobs(m),)),m,x)
expctd(m::NonlinearRegModel, x::Matrix{Float64}) = expctd!(Array(T,(nobs(m),)),m,x)

function expctdjac!{T<:Float64}(mu::Vector{T},jac::Matrix{T},m::OralSd1VClka,x::Vector{T})
    length(x) == 3 || error("length(x) = $(length(x)), should be 3")
    t = m.time; n = length(t); d = m.dose;
    V = x[1]
    Cl = x[2]
    ka = x[3]
    k = Cl/V                            # e2
    e1 = d/V
    e3 = ka - k
    e4 = ka / e3
    e5 = e1 * e4
    e14 = V * V
    e15 = Cl/e14
    e20 = e3 * e3
    e28 = 1./V
    for i in 1:n
        ti = t[i]
        e8 = exp(-k*ti)
        e11 = exp(-ka*ti)
        e12 = e8 - e11
        expctd[i] = e5*e12
        jac[i,1] = e5*(e8*(e15*ti)) - (e1*(ka*e15/e20) + d/e14*e4)*e12
        jac[i,2] = e1 * (ka * e28/e20) * e12 - e5 * (e8 * (e28 * ti))
        jac[i,3] = e1 * (1/e3 - ka/e20) * e12 + e5 * (e11 * ti)
    end
    mu, jac
end
const step = sqrt(eps())
const steps = [-step, step]
const mults = 1. + steps
function expctdjac(m::NLRegFD, x::Vector{Float64})
    p = length(x); pred = expctd(m, x); n = length(pred)
    jac = zeros(n,p)
    for j in 1:p
        par = copy(x); pjs = x[j] == 0. ? steps : x[j] * mults
        par[j] = pjs[2]
        rj = expctd(m, par)
        par[j] = pjs[1]
        jac[:,j] = (rj - expctd(m, par))/diff(pjs)
    end
    pred, jac
end
m = OralSd1VClka([0.25, 0.57, 1.12, 2.02, 3.82, 5.1, 7.03, 9.05, 12.12, 24.37], 4.02)

type NonlinearLS                    # nonlinear least squares problems
    pars::Vector{Float64}
    incr::Vector{Float64}
    obs::Vector{Float64}
    eta::Vector{Float64}                # expected response
    qtr::Vector{Float64}
    jacob::Matrix{Float64}              # Jacobian matrix
    qr::QR{Float64}
    m::NonlinearRegModel
    rss::Float64 # residual sum of squares at last successful iteration
    tolsqr::Float64    # squared tolerance for orthogonality criterion
    minfac::Float64
    mxiter::Int
    fit::Bool
end

function NonlinearLS(m::NonlinearRegModel, obs::Vector{Float64}, init::Vector{Float64})
    n = length(obs); p = length(init);
    eta, jacob = expctdjac(m, init)
    resid = obs - eta
    NonlinearLS(init, zero(init), obs, eta, resid, zero(resid), jacob,
                qrfact(jacob), m, sum(resid.^2), 1e-8, 0.5^10, 1000, false)
end
nlm = NonlinearLS(m, [2.84, 6.57, 10.5, 9.66, 8.58, 8.36, 7.47, 6.89, 5.94, 3.28], exp([-1,-4,0.55]))
# evaluate expected response and residual at m.pars + fac * m.incr
# return residual sum of squares
function updtres(nl::NonlinearLS, fac::Float64)
    expctd!(nl.m, nl.eta, nl.pars + fac * nl.incr)
    sqdiffsum(nl.obs,nl.eta)
end
    
# Create the QR factorization, qtr = Q'resid, solve for the increment and
# return the numerator of the squared convergence criterion
function qtr(nl::NonlinearLS)
    vs = nl.qr.vs; inc = nl.incr; qt = nl.qtr
    copy!(vs, nl.jacob); copy!(qt, nl.resid)
    _, T = geqrt3!(vs)
    copy!(nl.qr.T,T)
    gemqrt!('L','T',vs,T,qt)
    s = 0.; p = size(vs,2)
    for i in 1:p qti = qt[i]; s += qti * qti; inc[i] = qti  end
    trsv!('U','N','N',sub(vs,1:p,1:p),inc)
    s
end

function gnfit(nl::NonlinearLS)          # Gauss-Newton nonlinear least squares
    if !nl.fit
        converged = false; rss = nl.rss
        for i in 1:nl.mxiter
            crit = qtr(nl)/nl.rss # evaluate increment and orthogonality cvg. crit.
            converged = crit < nl.tolsqr
            f = 1.
            while f >= nl.minfac
                rss = updtres(nl,f)
                if rss < nl.rss break end
                f *= 0.5                    # step-halving
            end
            if f < nl.minfac
                error("Failure to reduce rss at $(nl.pars) with incr = $(nl.incr) and minfac = $(nl.minfact)")
            end
            nl.rss = rss
            nl.pars += f * nl.incr
            if converged break end
            nl.eta, nl.jacob = expctdjac(nl.m, nl.pars)  # evaluate Jacobian
        end
        if !converged error("failure to converge in $(nl.mxiter) iterations") end
        nl.fit = true
    end
    nl
end

function show(io::IO, nl::NonlinearLS)
    gnfit(nl)
    n,p = size(nl.jacob)
    s2 = nl.rss/float(n-p)
    varcov = s2 * symmetrize!(potri!('U', nl.qr.vs[1:p,:])[1],'U')
    stderr = sqrt(diag(varcov))
    t_vals = nl.pars./stderr
    println(io, "Model fit by nonlinear least squares to $n observations\n")
    println(io, DataFrame(parameter=nl.pars,stderr=stderr,t_value=nl.pars./stderr))
    println("Residual sum of squares at estimates = $(nl.rss)")
    println("Residual standard error = $(sqrt(s2)) on $(n-p) degrees of freedom")
end
