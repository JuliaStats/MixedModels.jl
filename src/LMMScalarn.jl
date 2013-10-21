type LMMScalarn{Ti<:Union(Int32,Int64)} <: LinearMixedModel
    A::CholmodSparse{Float64,Ti}
    L::CholmodFactor{Float64,Ti}
    Lambda::CholmodDense{Float64}
    Zt::CholmodSparse{Float64,Ti}
    ZtZ::CholmodSparse{Float64,Ti}
    RX::Cholesky{Float64}
    X::ModelMatrix{Float64}             # fixed-effects model matrix
    beta::Vector{Float64}
    fnms::Vector
    lambda::Vector{Float64}
    mu::Vector{Float64}
    offsets::Vector
    u::Vector{Vector{Float64}}
    y::Vector{Float64}
    REML::Bool
    fit::Bool
end

function Scalarn(X::ModelMatrix, Xs::Vector, facs::Vector,
                 y::Vector, fnms::Vector)
    refs = [f.refs for f in facs]; levs = [f.pool for f in facs]; k = length(Xs)
#    all([isnested(refs[i-1],refs[i]) for i in 2:k]) &&
#        return LMMNestedScalar(X,Xs,refs,levs,y,fnames)
    n,p = size(X); nlev = [length(l) for l in levs]; nz = hcat(Xs...)'
    offsets = [0, cumsum(nlev)]; q = offsets[end]
    Ti = q < typemax(Int32) ? Int32 : Int64
    rv = convert(Matrix{Ti}, broadcast(+, hcat(refs...)', offsets[1:k]))
    Zt = CholmodSparse!(convert(Vector{Ti}, [1:size(nz,1):length(nz)+1]),
                        copy(vec(rv)), vec(hcat(Xs...)'), q, n, 0)
    ZtZ = Zt*Zt'; L = cholfact(ZtZ,1.,true)
    u = Vector{Float64}[zeros(j) for j in nlev]
    LMMScalarn{Ti}(copy(ZtZ),L,CholmodDense!(ones(q)),Zt,ZtZ, cholfact(eye(p)),
        X,zeros(p),ones(k),ones(k),zeros(n),offsets,u,y,false,false)
end

##  cholfact(x, RX=true) -> the Cholesky factor of the downdated X'X or LambdatZt
cholfact(m::LMMScalarn,RX=true) = RX ? m.RX : m.L

## deviance!(m) -> Float64 : fit the model by maximum likelihood and return the deviance
deviance!(m::LMMScalarn) = objective(fit(reml!(m,false)))

##  grplevels(m) -> vector of number of levels in random-effect terms
grplevels(m::LMMScalarn) = [size(u,2) for u in m.u]

## isscalar(m) -> Bool : Are all the random-effects terms scalar?
isscalar(m::LMMScalarn) = true

## linpred!(m) -> update mu
function linpred!(m::LMMScalarn)
    gemv!('N',1.,m.X.m,m.beta,0.,m.mu)  # initialize mu to X*beta
    
    Xs = m.Xs; u = m.u; lm = m.lambda; inds = m.inds; mu = m.mu
    
    for i in 1:length(Xs)               # iterate over r.e. terms
        X = Xs[i]; ind = inds[i]
        if size(X,2) == 1 fma!(mu, (lm[i][1,1]*u[i])[:,ind], X[:,1])
        else
            add!(mu,sum(trmm('L','L','N','N',1.0,lm[i],u[i])[:,ind]' .* X, 2))
        end
    end
    m
end

## Logarithm of the determinant of the generator matrix for the Cholesky factor, L or RX
logdet(m::LMMScalarn,RX=true) = logdet(cholfact(m,RX))

## lower(m) -> lower bounds on elements of theta
lower(m::LMMScalarn) = [x==0.?-Inf:0. for x in vcat([ltri(eye(M)) for M in m.lambda]...)]

##  ranef(m) -> vector of matrices of random effects on the original scale
##  ranef(m,true) -> vector of matrices of random effects on the U scale
function ranef(m::LMMScalarn, uscale=false)
    uscale && return m.u
    Matrix{Float64}[m.lambda[i] * m.u[i] for i in 1:length(m.u)]
end

##  reml!(m,v=true) -> m : Set m.REML to v.  If m.REML is modified, unset m.fit
function reml!(m::LMMScalarn,v=true)
    v == m.REML && return m
    m.REML = v; m.fit = false
    m
end
    
## scale(m) -> estimate, s, of the scale parameter
## scale(m,true) -> estimate, s^2, of the squared scale parameter
function scale(m::LMMScalarn, sqr=false)
    n,p = size(m.X.m); ssqr = pwrss(m)/float64(n - (m.REML ? p : 0)); 
    sqr ? ssqr : sqrt(ssqr)
end

##  size(m) -> n, p, q, t (lengths of y, beta, u and # of re terms)
size(m::LMMScalarn) = (length(m.y), length(m.beta), m.offsets[end], length(m.u))

## solve!(m) -> m : solve PLS problem for u given beta
## solve!(m,true) -> m : solve PLS problem for u and beta
function solve!(m::LMMScalarn, ubeta=false)
    local u                             # so u from both branches is accessible
    n,p,q,k = size(m)
    if ubeta
        nzmat = reshape(m.LambdatZt.nzval, (div(length(m.LambdatZt.nzval),n),n))
        scrm = similar(nzmat); RZX = Array(Float64, sum(length, m.u), p)
        rvperm = m.rowvalperm
        cu = solve(m.L, cmult!(nzmat, m.y, scrm, RZX[:,1], rvperm), CHOLMOD_L)
        ttt = solve(m.L,cmult!(nzmat, m.X.m, scrm, RZX, rvperm),CHOLMOD_L)
        potrf!('U',syrk!('U','T',-1.,ttt,1.,syrk!('U','T',1.,m.X.m,0.,m.RX.UL)))
        potrs!('U',m.RX.UL,gemv!('T',-1.,ttt,cu,1.,gemv!('T',1.,m.X.m,m.y,0.,m.beta)))
        gemv!('N',-1.,ttt,m.beta,1.,cu)
        u = solve(m.L,solve(m.L,cu,CHOLMOD_Lt),CHOLMOD_Pt)
    else
        u = vec(solve(m.L,m.Lambda.mat .* m.Zt .* gemv!('N',-1.0,m.X.m,m.beta,1.0,copy(m.y))).mat)
    end
    pos = 0
    for i in 1:length(m.u)
        ll = length(m.u[i])
        m.u[i] = reshape(sub(u,pos+(1:ll)), size(m.u[i]))
        pos += ll
    end
    linpred!(m)
end

## sqrlenu(m) -> total squared length of m.u (the penalty in the PLS problem)
sqrlenu(m::LMMScalarn) = sum([mapreduce(Abs2Fun(),Add(),u) for u in m.u])

## std(m) -> Vector{Vector{Float64}} estimated standard deviations of variance components
std(m::LMMScalarn) = scale(m)*push!(Vector{Float64}[vec(vnorm(l,2,1)) for l in m.lambda],[1.])

## theta(m) -> vector of variance-component parameters
theta(m::LMMScalarn) = m.lambda

##  theta!(m,th) -> m : install new value of theta, update LambdatZt and L 
function theta!(m::LMMScalarn, th::Vector{Float64})
    off = m.offsets; k = length(off) - 1
    length(th) == k || error("Dimension mismatch")
    A = m.A; copy!(A.nzval, m.ZtZ.nzval); lam = m.Lambda.mat
    for i in 1:k
        thi = th[i]
        for j in (off[i]+1):off[i+1]
            lam[j,1] = thi
        end
    end
    cholfact!(m.L,chm_scale!(m.A,m.Lambda,CHOLMOD_SYM),1.)
    m
end
