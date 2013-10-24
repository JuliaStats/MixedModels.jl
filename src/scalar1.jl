## The ScalarLMM1 type represents a model with a single scalar random-effects term

## Fields are arranged by decreasing size, doubles then pointers then bools
type LMMScalar1 <: LinearMixedModel
    theta::Float64
    L::Vector{Float64}
    RX::Cholesky{Float64}
    Xt::Matrix{Float64}
    XtX::Matrix{Float64}
    XtZ::Matrix{Float64}
    Xty::Vector{Float64}
    Ztrv::Vector                        # indices into factor levels
    Ztnz::Vector{Float64}               # left-hand side of r.e. term
    ZtZ::Vector{Float64}
    Zty::Vector{Float64}
    beta::Vector{Float64}
    fname::String
    mu::Vector{Float64}
    u::Vector{Float64}
    y::Vector
    REML::Bool
    fit::Bool
end

function LMMScalar1(X::ModelMatrix, Xs::Matrix, fac::PooledDataArray,
                    y::Vector, fname::String)
    Xt = X.m'; p,n = size(Xt); Ztnz = vec(Xs); q = length(fac.pool); rv = fac.refs
    n == length(rv) == length(Ztnz) == length(y) || error("Dimension mismatch")
    XtX = Xt*Xt'; ZtZ = zeros(q); XtZ = zeros(p,q); Zty = zeros(q);
    for i in 1:n
        j = rv[i]; z = Ztnz[i]
        ZtZ[j] += abs2(z); Zty[j] += z*y[i]; XtZ[:,j] += z*Xt[:,i]
    end
    LMMScalar1(1., ones(q), cholfact(XtX), Xt, XtX, XtZ, Xt*y, rv, Ztnz , ZtZ, Zty,
               zeros(p), fname, zeros(n), zeros(q), copy(y), false, false)
end

##  cholfact(x, RX=true) -> the Cholesky factor of the downdated X'X or LambdatZt
cholfact(m::LMMScalar1,RX=true) = RX ? m.RX : Diagonal(m.L)

fnames(m::LMMScalar1) = String[fname]

grplevels(m::LMMScalar1) = [length(m.u)]

isscalar(m::LMMScalar1) = true

## linpred!(m) -> m   -- update mu
function linpred!(m::LMMScalar1)
    for i in 1:length(m.mu)             # mu = Z*Lambda*u
        m.mu[i] = m.theta * m.u[m.Ztrv[i]] * m.Ztnz[i]
    end
    gemv!('T',1.,m.Xt,m.beta,1.,m.mu)   # mu += X'beta
    m
end

## Logarithm of the determinant of the generator matrix for the Cholesky factor, RX or L
logdet(m::LMMScalar1,RX=true) = RX ? logdet(m.RX) : 2.sum(LogFun(),m.L)

## lower(m) -> lower bounds on elements of theta
lower(m::LMMScalar1) = zeros(1)

##  ranef(m) -> vector of matrices of random effects on the original scale
##  ranef(m,true) -> vector of matrices of random effects on the U scale
function ranef(m::LMMScalar1, uscale=false)
    uscale && return [m.u']
    [m.theta * m.u']
end

##  size(m) -> n, p, q, t (lengths of y, beta, u and # of re terms)
size(m::LMMScalar1) = (length(m.y), length(m.beta), length(m.u), 1)

## sqrlenu(m) -> total squared length of m.u (the penalty in the PLS problem)
sqrlenu(m::LMMScalar1) = sumsq(m.u)

theta(m::LMMScalar1) = [m.theta]

##  theta!(m,th) -> m : install new value of theta, update LambdatZt and L 
function theta!(m::LMMScalar1, th::Vector{Float64})
    length(th) == 1 || error("LMMScalar1 theta must have length 1")
    m.theta = th[1]; n,p,q,t = size(m)
    m.theta >= 0. || error("theta = $th must be >= 0")
    map!(x->sqrt(m.theta*m.theta*x + 1.), m.L, m.ZtZ)
    m
end

## solve!(m) -> m : solve PLS problem for u given beta
## solve!(m,true) -> m : solve PLS problem for u and beta
function solve!(m::LMMScalar1, ubeta=false)
    thlinv = m.theta / m.L
    map!(Multiply(), m.u, m.Zty, thlinv) # initialize u to cu
    if ubeta
        LXZ = scale(m.XtZ, thlinv)
        potrf!('U', syrk!('U', 'N', -1., LXZ, 1., copy!(m.RX.UL, m.XtX)))
        copy!(m.beta, m.Xty)              # initialize beta to Xty
        gemv!('N',-1.,LXZ,m.u,1.,m.beta)  # cbeta = Xty - RZX'cu
        solve!(m.RX, m.beta)              # solve for beta in place
        gemv!('T',-1.,LXZ,m.beta,1.,m.u)  # cu -= RZX'beta
    end
    map1!(Divide(), m.u, m.L)           # solve for u in place
    linpred!(m)                         # update mu
end

## std(m) -> Vector{Vector{Float64}} estimated standard deviations of variance components
std(m::LMMScalar1) = scale(m)*[m.theta, 1.]
