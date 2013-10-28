type LMMGeneral{Ti<:Union(Int32,Int64)} <: LinearMixedModel
    L::CholmodFactor{Float64,Ti}
    LambdatZt::CholmodSparse{Float64,Ti}
    RX::Cholesky{Float64}
    X::ModelMatrix{Float64}             # fixed-effects model matrix
    Xs::Vector{Matrix{Float64}}         # X_1,X_2,...,X_k
    XtX::Symmetric{Float64}
    Xty::Vector{Float64}
    beta::Vector{Float64}
    inds::Vector
    lambda::Vector{Matrix{Float64}}     # k lower triangular mats
    mu::Vector{Float64}
    perm::Vector{Ti}
    pvec::Vector
    u::Vector{Matrix{Float64}}
    y::Vector{Float64}
    REML::Bool
    fit::Bool
end

## function LMMGeneral(X::ModelMatrix, Xs::Vector, facs::Vector,
##                     y::Vector, fnms::Vector, pvec::Vector)
##     refs = [f.refs for f in facs]; levs = [f.pool for f in facs]; k = length(Xs)
##     n,p = size(X); nlev = [length(l) for l in levs]; nz = hcat(Xs...)'
##     nu = nlev .* pvec; offsets = [0,cumsum(nu)]; q = offsets[end]
##     Ti = q > typemax(Int32) ? Int64 : Int32
##     rv = convert(Matrix{Ti}, broadcast(+, hcat(refs...)', offsets[1:k]))
##     LambdatZt = CholmodSparse!(convert(Vector{Ti}, [1:size(nz,1):length(nz)+1]),
##                                vec(copy(rv)), vec(nz), q, n, 0)
##     L = cholfact(LambdatZt,1.,true)
##     LMMGeneral{Ti}(L,LambdatZt,cholfact(eye(p)),X,Xs,zeros(p),inds,lambda,
##         zeros(n),L.Perm + one(Ti),u,y,false,false)
## end

##  cholfact(x, RX=true) -> the Cholesky factor of the downdated X'X or LambdatZt
cholfact(m::LMMGeneral,RX=true) = RX ? m.RX : m.L

## cor(m) -> correlation matrices of variance components
cor(m::LMMGeneral) = [cc(l) for l in m.lambda]

## deviance!(m) -> Float64 : fit the model by maximum likelihood and return the deviance
deviance!(m::LMMGeneral) = objective(fit(reml!(m,false)))

##  grplevels(m) -> vector of number of levels in random-effect terms
grplevels(m::LMMGeneral) = [size(u,2) for u in m.u]

## isscalar(m) -> Bool : Are all the random-effects terms scalar?
isscalar(m::LMMGeneral) = all(pvec .== 1)

## linpred!(m) -> update mu
function linpred!(m::LMMGeneral)
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
logdet(m::LMMGeneral,RX=true) = logdet(cholfact(m,RX))

## lower(m) -> lower bounds on elements of theta
lower(m::LMMGeneral) = vcat([lower_bd_ltri(p) for p in m.pvec]...)

##  ranef(m) -> vector of matrices of random effects on the original scale
##  ranef(m,true) -> vector of matrices of random effects on the U scale
function ranef(m::LMMGeneral, uscale=false)
    uscale && return m.u
    Matrix{Float64}[m.lambda[i] * m.u[i] for i in 1:length(m.u)]
end

## solve!(m) -> m : solve PLS problem for u given beta
## solve!(m,true) -> m : solve PLS problem for u and beta
function solve!(m::LMMGeneral, ubeta=false)
    local u                             # so u from both branches is accessible
    n,p,q,k = size(m)
    if ubeta
        cu = solve(m.L,permute!(vec((m.LambdatZt * m.y).mat),m.perm),CHOLMOD_L)
        RZX = (m.LambdatZt * m.X.m).mat
        for j in 1:size(RZX,2)
            permute!(sub(RZX,:,j),m.perm)
        end
        RZX = solve(m.L, RZX, CHOLMOD_L)
        _,info = potrf!('U',syrk!('U','T',-1.,RZX,1.,copy!(m.RX.UL,m.XtX.S)))
        info == 0 || error("downdated X'X is singular")
        potrs!('U',m.RX.UL,gemv!('T',-1.,RZX,cu,1.,copy!(m.beta,m.Xty)))
        gemv!('N',-1.,RZX,m.beta,1.,cu)
        u = ipermute!(solve(m.L,gemv!('N',-1.,RZX,m.beta,1.,cu),CHOLMOD_Lt),m.perm)
    else
        u = vec(solve(m.L,m.LambdatZt * gemv!('N',-1.0,m.X.m,m.beta,1.0,copy(m.y))).mat)
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
sqrlenu(m::LMMGeneral) = sum([mapreduce(Abs2Fun(),Add(),u) for u in m.u])

## std(m) -> Vector{Vector{Float64}} estimated standard deviations of variance components
std(m::LMMGeneral) = scale(m)*push!(Vector{Float64}[vec(vnorm(l,2,1)) for l in m.lambda],[1.])

## theta(m) -> vector of variance-component parameters
theta(m::LMMGeneral) = vcat([ltri(M) for M in m.lambda]...)

##  theta!(m,th) -> m : install new value of theta, update LambdatZt and L 
function theta!(m::LMMGeneral, th::Vector{Float64})
    n = length(m.y)
    nzmat = reshape(m.LambdatZt.nzval, (div(length(m.LambdatZt.nzval),n),n))
    lambda = m.lambda; Xs = m.Xs; tpos = 1; roff = 0 # position in th, row offset
    for kk in 1:length(Xs)
        T = lambda[kk]; p = size(T,1) # size of i'th template matrix
        for j in 1:p, i in j:p        # fill lower triangle from th
            T[i,j] = th[tpos]; tpos += 1
            i == j && T[i,j] < 0. && error("Negative diagonal element in T")
        end
        gemm!('T','T',1.,T,Xs[kk],0.,sub(nzmat,roff+(1:p),1:n))
        roff += p
    end
    cholfact!(m.L,m.LambdatZt,1.)
    m
end
