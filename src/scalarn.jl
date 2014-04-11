type LMMScalarn{Ti<:Union(Int32,Int64)} <: LinearMixedModel
    A::CholmodSparse{Float64,Ti}
    L::CholmodFactor{Float64,Ti}
    Lambda::Diagonal
    RX::Base.LinAlg.Cholesky{Float64}
    X::ModelMatrix{Float64}             # fixed-effects model matrix
    Xty::Vector{Float64}
    Zt::SparseMatrixCSC{Float64,Ti}
    ZtX::Matrix{Float64}
    ZtZ::CholmodSparse{Float64,Ti}
    Zty::Vector{Float64}
    beta::Vector{Float64}
    fnms::Vector
    lambda::Vector{Float64}
    mu::Vector{Float64}
    offsets::Vector
    perm::Vector                        # fill-reducing permutation
    u::Vector{Vector{Float64}}
    y::Vector
    REML::Bool
    fit::Bool
end

##  cholfact(x, RX=true) -> the Cholesky factor of the downdated X'X or LambdatZt
cholfact(m::LMMScalarn,RX=true) = RX ? m.RX : m.L

## deviance!(m) -> Float64 : fit the model by maximum likelihood and return the deviance
deviance!(m::LMMScalarn) = objective(fit(reml!(m,false)))

##  grplevels(m) -> vector of number of levels in random-effect terms
grplevels(m::LMMScalarn) = [length(u) for u in m.u]

## isscalar(m) -> Bool : Are all the random-effects terms scalar?
isscalar(m::LMMScalarn) = true

## linpred!(m) -> update mu
function linpred!(m::LMMScalarn)
    gemv!('N',1.,m.X.m,m.beta,1.,At_mul_B!(1.,m.Zt,m.Lambda*vcat(m.u...),0.,m.mu))
    m
end

## Logarithm of the determinant of the generator matrix for the Cholesky factor, L or RX
logdet(m::LMMScalarn,RX=true) = logdet(cholfact(m,RX))

## lower(m) -> lower bounds on elements of theta
lower(m::LMMScalarn) = zeros(length(m.fnms))

##  ranef(m) -> vector of matrices of random effects on the original scale
##  ranef(m,true) -> vector of matrices of random effects on the U scale
function ranef(m::LMMScalarn, uscale=false)
    uscale && return m.u
    map(.*,m.lambda,m.u)
end
    
## scale(m) -> estimate, s, of the scale parameter
## scale(m,true) -> estimate, s^2, of the squared scale parameter
function scale(m::LMMScalarn, sqr=false)
    n,p = size(m.X); ssqr = pwrss(m)/float64(n - (m.REML ? p : 0)); 
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
        cu = solve(m.L, permute!(m.Lambda * m.Zty, m.perm), CHOLMOD_L)
        RZX = solve(m.L, (m.Lambda * m.ZtX)[m.perm,:], CHOLMOD_L)
        cholfact!(syrk!('U','T',-1.,RZX,1.,syrk!('U','T',1.,m.X.m,0.,m.RX.UL)))
        A_ldiv_B!(m.RX,gemv!('T',-1.,RZX,cu,1.,copy!(m.beta,m.Xty)))
        u = ipermute!(solve(m.L,gemv!('N',-1.,RZX,m.beta,1.,cu),CHOLMOD_Lt),m.perm)
    else
        u = m.L\ (m.Lambda * (m.Zt*gemv!('N',-1.0,m.X.m,m.beta,1.0,copy(m.y))))
    end
    pos = 0
    for i in 1:length(m.u)
        ui = m.u[i]
        for j in 1:length(ui) ui[j] = u[pos += 1] end
    end
    linpred!(m)
end

## sqrlenu(m) -> total squared length of m.u (the penalty in the PLS problem)
sqrlenu(m::LMMScalarn) = sum([sumsq(u) for u in m.u])

## std(m) -> Vector{Vector{Float64}} estimated standard deviations of variance components
std(m::LMMScalarn) = scale(m)*[m.lambda,1.]

## theta(m) -> vector of variance-component parameters
theta(m::LMMScalarn) = m.lambda

##  theta!(m,th) -> m : install new value of theta, update LambdatZt and L 
function theta!(m::LMMScalarn, th::Vector{Float64})
    off = m.offsets; k = length(off) - 1
    length(th) == k || error("Dimension mismatch")
    copy!(m.lambda,th)
    A = m.A; copy!(A.nzval, m.ZtZ.nzval); lam = m.Lambda.diag
    for i in 1:k
        thi = th[i]
        for j in (off[i]+1):off[i+1]
            lam[j] = thi
        end
    end
    cholfact!(m.L,chm_scale!(m.A,m.Lambda.diag,CHOLMOD_SYM),1.)
    m
end
