## A linear mixed model with a single vector-valued random-effects term

type LMMVector1 <: LinearMixedModel
    lmb::LMMBase
    ldL2::Float64
    L::Array{Float64,3}
    RX::Base.LinAlg.Cholesky{Float64}
    RZX::Array{Float64,3}
    Xt::Matrix{Float64}
    XtX::Matrix{Float64}
    Xty::Vector{Float64}
    Ztrv::Vector
    Ztnz::Matrix{Float64}
    ZtX::Array{Float64,3}
    ZtZ::Array{Float64,3}
    Zty::Matrix{Float64}
    beta::Vector{Float64}
    lambda::Matrix{Float64}
    u::Matrix{Float64}
    REML::Bool
    fit::Bool
end

function LMMVector1(lmb::LMMBase)
    Xt = lmb.X.m'; p,n = size(Xt); Ztnz = lmb.Xs[1]'; grp = lmb.facs[1];
    rv = grp.refs; nl = length(grp.pool); y = lmb.y; k = size(Ztnz,1)
    XtX = Xt*Xt'; ZtZ = zeros(k,k,nl); ZtX = zeros(k,p,nl); Zty  = zeros(k,nl)
    for j in 1:n
        i = rv[j]; z = Ztnz[:,j]; ZtZ[:,:,i] += z*z';
        Zty[:,i] += y[j]*z; ZtX[:,:,i] += z*Xt[:,j]'
    end
    LMMVector1(lmb, 0., zeros(k,k,nl), cholfact(XtX,:U), similar(ZtX), Xt, XtX,
               Xt*y, rv, Ztnz, ZtX, ZtZ, Zty, zeros(p),
               eye(k), zeros(k,nl), false, false)
end

Base.cholfact(m::LMMVector1,RX=true) = RX ? m.RX : error("not yet written")

## cor(m) -> correlation matrices of variance components
Base.cor(m::LMMVector1) = [cc(m.lambda)]

## fit(m) -> m Optimize the objective using MMA from the NLopt package
function StatsBase.fit(m::LMMVector1, verbose=false)
    if !isfit(m)
        th = theta(m); k = length(th)
        opt = Opt(:LD_MMA, k)
        ftol_abs!(opt, 1e-6)    # criterion on deviance changes
        ftol_rel!(opt, 1e-9)
        xtol_abs!(opt, 1e-6)    # criterion on all parameter value changes
        lower_bounds!(opt, lower(m))
        function obj(x::Vector{Float64}, g::Vector{Float64})
            rr = objective(solve!(theta!(m,x),true))
            if length(g) > 0
                copy!(g, grad(m))
            end
            rr
        end
        if verbose
            count = 0
            function vobj(x::Vector{Float64}, g::Vector{Float64})
                count += 1
                val = objective(solve!(theta!(m,x),true))
                print("f_$count: $(round(val,5)), "); showcompact(x); println()
                if length(g) > 0
                    copy!(g, grad(m))
                end
                val
            end
            min_objective!(opt, vobj)
        else
            min_objective!(opt, obj)
        end
        fmin, xmin, ret = optimize(opt, th)
        if verbose println(ret) end
        m.fit = true
    end
    m
end

function grad(m::LMMVector1)        # called after solve!
    n,p,q = size(m); k,nl = size(m.u); L = m.L; ZtZ = m.ZtZ; lambda = m.lambda
    mu = m.lmb.mu; rv = m.Ztrv; nz = m.Ztnz; res = zeros(k,k)
    for i in 1:nl
        res += Base.LinAlg.LAPACK.potrs!('L',sub(L,:,:,i), lambda'*sub(ZtZ,:,:,i))
    end
    Ztr = copy(m.Zty)          # create Z'(resid) starting with Zty
    for i in 1:n Ztr[:,rv[i]] -= mu[i] * nz[:,i] end
    ltri(BLAS.syr2k!('L','N',-1./scale(m,true),Ztr,m.u,1.,res+res'))
end

## linpred!(m) -> m   -- update mu
function linpred!(m::LMMVector1)
    mu = m.lmb.mu
    BLAS.gemv!('T',1.,m.Xt,m.beta,0.,mu) # initialize to X*beta
    bb = BLAS.trmm('L','L','N','N',1.,m.lambda,m.u) # b = Lambda * u
    k = size(bb,1)
    for i in 1:length(mu)
        mu[i] += dot(sub(bb,1:k,int(m.Ztrv[i])), sub(m.Ztnz,1:k,i)) end
    m
end

## Logarithm of the determinant of the generator matrix for the Cholesky factor, RX or L
Base.logdet(m::LMMVector1,RX=true) = RX ? logdet(m.RX) : m.ldL2
    
lower(m::LMMVector1) = lower_bd_ltri(size(m.Ztnz,1))

##  ranef(m) -> vector of matrices of random effects on the original scale
##  ranef(m,true) -> vector of matrices of random effects on the U scale
function ranef(m::LMMVector1, uscale=false)
    uscale && return [m.u]
    [BLAS.trmm('L','L','N','N',1.,m.lambda,m.u)]
end
    
Base.size(m::LMMVector1) = (length(m.lmb.y), length(m.beta), length(m.u), 1)

## solve!(m) -> m : solve PLS problem for u given beta
## solve!(m,true) -> m : solve PLS problem for u and beta
function solve!(m::LMMVector1, ubeta=false)
    n,p,q = size(m); k,nl = size(m.u); copy!(m.u,m.Zty)
    BLAS.trmm!('L','L','T','N',1.,m.lambda,m.u)
    for l in 1:nl                       # cu := L^{-1} Lambda'Z'y
        BLAS.trsv!('L','N','N',sub(m.L,1:k,1:k,l),sub(m.u,1:k,l))
    end
    if ubeta
        copy!(m.beta,m.Xty); copy!(m.RZX,m.ZtX); copy!(m.RX.UL, m.XtX)
        BLAS.trmm!('L','L','T','N',1.,m.lambda,reshape(m.RZX,(k,p*nl))) # Lambda'Z'X
        for l in 1:nl
            wL = sub(m.L,1:k,1:k,l); wRZX = sub(m.RZX,1:k,1:p,l)
            BLAS.trsm!('L','L','N','N',1.,wL,wRZX) # solve for l'th face of RZX
            BLAS.gemv!('T',-1.,wRZX,sub(m.u,1:k,l),1.,m.beta) # downdate m.beta
            BLAS.syrk!('U','T',-1.,wRZX,1.,m.RX.UL)           # downdate XtX
        end
        _, info = LAPACK.potrf!('U',m.RX.UL) # Cholesky factor RX
        bool(info) && error("Downdated X'X is not positive definite")
        A_ldiv_B!(m.RX,m.beta)           # beta = (RX'RX)\(downdated X'y)
        for l in 1:nl                 # downdate cu
            BLAS.gemv!('N',-1.,sub(m.RZX,1:k,1:p,l),m.beta,1.,sub(m.u,1:k,l))
        end
    end
    for l in 1:nl                     # solve for m.u
        BLAS.trsv!('L','T','N',sub(m.L,1:k,1:k,l), sub(m.u,1:k,l))
    end
    linpred!(m)
end        

## sqrlenu(m) -> total squared length of m.u (the penalty in the PLS problem)
sqrlenu(m::LMMVector1) = NumericExtensions.sumsq(m.u)

## std(m) -> Vector{Vector{Float64}} estimated standard deviations of variance components
Base.std(m::LMMVector1) = scale(m)*push!(copy(vec(vnorm(m.lambda,2,1))),1.)

theta(m::LMMVector1) = ltri(m.lambda)

##  theta!(m,th) -> m : install new value of theta, update L 
function theta!(m::LMMVector1, th::Vector{Float64})
    n,p,q,t = size(m); k,nl = size(m.u); pos = 1
    for j in 1:k, i in j:k
        m.lambda[i,j] = th[pos]; pos += 1
    end
    ldL = 0.; copy!(m.L,m.ZtZ)
    BLAS.trmm!('L','L','T','N',1.,m.lambda,reshape(m.L,(k,q)))
    for l in 1:nl
        wL = sub(m.L,1:k,1:k,l)
        BLAS.trmm!('R','L','N','N',1.,m.lambda,wL) # lambda'(Z'Z)_l*lambda
        for j in 1:k                    # Inflate the diagonal
            wL[j,j] += 1.
        end
        _, info = LAPACK.potrf!('L',wL) # i'th diagonal block of L_Z
        bool(info) && error("Cholesky decomposition failed at l = $l")
        for j in 1:k ldL += log(wL[j,j]) end
    end
    m.ldL2 = 2.ldL
    m
end
