abstract Delta         # An arrow shaped A/L block from nested factors

type DeltaLeaf <: Delta   # A terminal Delta block
    Ad::Array{Float64,3}                # diagonal blocks
    Ab::Array{Float64,3}                # base blocks
    At::Symmetric{Float64}              # lower right block
    Ld::Array{Float64,3}                # diagonal blocks
    Lb::Array{Float64,3}                # base blocks
    Lt::Base.LinAlg.Cholesky{Float64}   # lower right triangle
end

function DeltaLeaf(Ad::Array{Float64,3}, Ab::Array{Float64,3}, At::Symmetric{Float64})
    m,n,t = size(Ad)
    m == n || error("Faces of Ad must be square")
    p,q,r = size(Ab)
    p == size(At,1) && q == n && r == t || error("Size mismatch")
    DeltaLeaf(Ad,Ab,At,zeros(Ad),zeros(Ab),cholfact(At.S,symbol(At.uplo)))
end

function DeltaLeaf(Ad::Array{Float64,3}, Ab::Array{Float64,3}, At::Matrix{Float64})
    DeltaLeaf(Ad,Ab,Symmetric(At,:L))
end

function DeltaLeaf(ff::PooledDataVector, Xst::Matrix, Xt::Matrix)
    refs = ff.refs
    (L = length(refs)) == size(Xst,2) == size(Xt,2) || error("Dimension mismatch")
    m = size(Xt,1)
    n = size(Xst,1)
    nl = length(ff.pool)         # number of levels of grouping factor
    Ad = zeros(n,n,nl)
    Ab = zeros(m,n,nl)
    for j in 1:L
        jj = int(refs[j])
        BLAS.syr!('L',1.0,sub(Xst,:,j),sub(Ad,:,:,jj))
        BLAS.gemm!('N','T',1.0,sub(Xst,:,j:j),sub(Xt,:,j:j),1.0,sub(Ab,:,:,jj))
    end
    DeltaLeaf(Ad,Ab,Symmetric(Xt*Xt',:L))
end

function DeltaLeaf(lmb::LMMBase)
    length(lmb.facs) == 1 || error("DeltaLeaf requires a single r.e. term")
    DeltaLeaf(lmb.facs[1],lmb.Xs[1]',lmb.X.m')
end

## Logarithm of the determinant of the generator matrix for the Cholesky factor, RX or L
function Base.logdet(dl::DeltaLeaf,RX=true)
    RX && return logdet(dl.Lt)
    Ld = dl.Ld
    m,n,t = size(Ld)
    s = 0.
    @inbounds for j in 1:t, i in 1:m
        s += log(Ld[i,i,j])
    end
    2.s
end

Base.size(dl::DeltaLeaf) = size(dl.Ab)

function Base.Triangular{T<:Number}(n::Integer,v::Vector{T})
    length(v) == n*(n+1)>>1 || error("Dimension mismatch")
    A = zeros(T,n,n)
    pos = 0
    for j in 1:n, i in j:n
        A[i,j] = v[pos += 1]
    end
    Triangular(A,:L,false)
end

##  updateL!(dl,Lambda)
function updateL!(dl::DeltaLeaf,lambda::Triangular{Float64})
    m,n,l = size(dl)
    size(lambda,1) == n || error("Dimension mismatch")
    if n == 1
        dl.Ld[:] = sqrt(abs2(lambda[1,1])*vec(dl.Ad) .+ 1.)
    else
        Ld = dl.Ld
        copy!(Ld,dl.Ad)
        Ac_mul_B!(lambda,reshape(Ld,(n,n*l)))
        for k in 1:l
            (wL = sub(Ld,:,:,k)) * lambda
            for j in 1:n                # Inflate the diagonal
                wL[j,j] += 1.
            end
            _, info = LAPACK.potrf!('L',wL) # i'th diagonal block of L_Z
            bool(info) && error("Cholesky decomposition failed at k = $k")
        end
    end
end

## solve!(dl,u) -> m : solve PLS problem for u given beta
## solve!(dl,u,beta) -> m : solve PLS problem for u and beta
function solve!(dl:DeltaLeaf, u::Matrix{Float64}, beta=Float64[])
    m,n,l = size(dl)
    (n,l) == size(u) || error("Dimension mismatch")
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

type DeltaNode <: Delta
    Ad::Vector{Delta}
    Ab::Array{Float64,3}
    At::Symmetric{Float64}
    Lt::Base.LinAlg.Cholesky{Float64}
end
