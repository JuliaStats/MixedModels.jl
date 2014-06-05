type GenSolver <: PLSSolver
    L::CholmodFactor{Float64,Ti}
    RX::Base.LinAlg.Cholesky{Float64}
    RZX::Matrix{Float64}
    XtX::Symmetric{Float64}
    ZtX::Matrix{Float64}
    ZtZ::CholmodSparse{Float64,Ti}
    perm::Vector{Ti}
end

function GenSolver(lmb::LMMBase)        # incomplete
    zt = λtZt(lmb)
    Ztc = CholmodSparse(zt)
    ZtZ = Ztc * Ztc'
    L = cholfact(ZtZ,1.,true)
    perm = L.Perm
    X = lmb.X.m
    XtX = Symmetric(X'X,:L)
    ZtX = zt*X
    GenSolver(L,cholfact(XtX.S,:L),copy(ZtX),XtX,ZtX,XtX,perm+one(eltype(perm)))
end

LMMGeneral(lmb::LMMBase) = LMMGeneral(lmb,λtZtSolver)

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
Base.cholfact(m::LMMGeneral,RX=true) = RX ? m.RX : m.L

## cor(m) -> correlation matrices of variance components
Base.cor(m::LMMGeneral) = [cc(l) for l in m.lambda]

## deviance!(m) -> Float64 : fit the model by maximum likelihood and return the deviance
deviance!(m::LMMGeneral) = objective(fit(reml!(m,false)))

## linpred!(m) -> update mu
function linpred!(m::LMMGeneral)
    BLAS.gemv!('N',1.,m.X.m,m.beta,0.,m.mu)  # initialize mu to X*beta
    Xs = m.Xs; u = m.u; lm = m.lambda; inds = m.inds; mu = m.mu
    for i in 1:length(Xs)               # iterate over r.e. terms
        X = Xs[i]
        ind = inds[i]
        if size(X,2) == 1
            fma!(mu, (lm[i][1,1]*u[i])[:,ind], X[:,1])
        else
            add!(mu,sum(BLAS.trmm('L','L','N','N',1.0,lm[i],u[i])[:,ind]' .* X, 2))
        end
    end
    m
end

## Logarithm of the determinant of the generator matrix for the Cholesky factor, L or RX
Base.logdet(m::LMMGeneral,RX=true) = logdet(cholfact(m,RX))

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
        cu = solve(m.L,permute!(m.LambdatZt * m.y,m.perm),CHOLMOD_L)
        RZX = m.LambdatZt * m.X.m
        for j in 1:size(RZX,2)
            permute!(sub(RZX,:,j),m.perm) # needs view instead of sub?
        end
        RZX = solve(m.L, RZX, CHOLMOD_L)
        _,info = LAPACK.potrf!('U',BLAS.syrk!('U','T',-1.,RZX,1.,copy!(m.RX.UL,m.XtX.S)))
        info == 0 || error("downdated X'X is singular")
        LAPACK.potrs!('U',m.RX.UL,BLAS.gemv!('T',-1.,RZX,cu,1.,copy!(m.beta,m.Xty)))
        u = ipermute!(solve(m.L,BLAS.gemv!('N',-1.,RZX,m.beta,1.,cu),CHOLMOD_Lt),m.perm)
    else
        u = vec(solve(m.L,m.LambdatZt * BLAS.gemv!('N',-1.0,m.X.m,m.beta,1.0,copy(m.y))).mat)
    end
    pos = 0
    for i in 1:length(m.u)
        ll = length(m.u[i])
        m.u[i] = reshape(sub(u,pos+(1:ll)), size(m.u[i]))
        pos += ll
    end
    linpred!(m)
end

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
        BLAS.gemm!('T','T',1.,T,Xs[kk],0.,sub(nzmat,roff+(1:p),1:n))
        roff += p
    end
    cholfact!(m.L,m.LambdatZt,1.)
    m
end
