abstract PLSSolver

abstract Delta <: PLSSolver # An arrow shaped A/L block from nested factors

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
    for j in 1:nl        # symmetrize the faces created with BLAS.syr!
        Base.LinAlg.copytri!(sub(Ad,:,:,j),'L')
    end
    DeltaLeaf(Ad,Ab,Symmetric(Xt*Xt',:L))
end

function DeltaLeaf(lmb::LMMBase)
    length(lmb.facs) == 1 || error("DeltaLeaf requires a single r.e. term")
    DeltaLeaf(lmb.facs[1],lmb.Xs[1],lmb.X.m')
end

Base.cholfact(dl::DeltaLeaf,RX::Bool=true) = RX ? dl.Lt : error("Code not yet written")

## Logarithm of the determinant of the matrix represented by, RX or L
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
Base.size(dl::DeltaLeaf,k::Integer) = size(dl.Ab,k)

function Base.Triangular{T<:Number}(n::Integer,v::Vector{T})
    length(v) == n*(n+1)>>1 || error("Dimension mismatch")
    A = zeros(T,n,n)
    pos = 0
    for j in 1:n, i in j:n
        A[i,j] = v[pos += 1]
    end
    Triangular(A,:L,false)
end

##  update!(dl,lambda)->dl : update Ld, Lb and Lt
function update!(dl::DeltaLeaf,λ::Triangular)
    m,n,l = size(dl)
    n == size(λ,1) || error("Dimension mismatch")
    Lt = copy!(dl.Lt.UL,dl.At.S)
    if n == 1                           # shortcut for 1×1 λ
        lam = λ[1,1]
        Ld = map!(x -> sqrt(x*lam*lam + 1.), dl.Ld, dl.Ad)
        Lb = reshape(copy!(dl.Lb,dl.Ab),(m,m*l))
        Lt[1,1] -= Base.sumabs2(scale!(scale!(lam,Lb),1.0 ./ vec(Ld)))
    else
        BLAS.trmm!('L',λ.uplo,'T',λ.unitdiag,1.,λ.UL,reshape(copy!(dl.Ld,dl.Ad),(n,n*l)))
        Lb = BLAS.trmm!('L',λ.uplo,'T',λ.unitdiag,1.,λ.UL,reshape(copy!(dl.Lb,dl.Ab),(m,n*l)))
        for k in 1:l
            wL = BLAS.trmm!('R',λ.uplo,'N',λ.unitdiag,1.,λ.UL,sub(dl.Ld,:,:,k))
            for j in 1:n                # Inflate the diagonal
                wL[j,j] += 1.
            end
            _, info = LAPACK.potrf!('L',wL) # i'th diagonal block of L
            info == 0 || error("Cholesky failure at L diagonal block $k")
            wRZX = BLAS.trsm!('L','L','N','N',1.,wL,sub(dl.Lb,:,:,k))
            BLAS.syrk!('L','T',-1.0,wRZX,1.,Lt)
        end
    end
    _, info = LAPACK.potrf!('L',Lt)
    info == 0 ||  error("downdated X'X is not positive definite")
    dl
end

function update!(dl::DeltaLeaf,λ::Vector)
    length(λ) == 1 || error("update! on a DeltaLeaf requires length(λ) == 1")
    update!(dl,λ[1])
end

function Base.A_ldiv_B!(dl::DeltaLeaf,lmb::LMMBase)
    n,p,q,l = size(lmb)
    l == 1 || error("DeltaLeaf should take an LMMBase with 1 r.e. term")
    λ = lmb.λ[1]
    cu = BLAS.trmm!('L',λ.uplo,'T',λ.unitdiag,1.,λ.UL,copy!(lmb.u[1],lmb.Zty[1]))
    β = copy!(lmb.β,lmb.Xty)
    if size(λ,1) == 1                   # short cut for scalar r.e.
        Linv = 1. ./ vec(dl.Ld)
        scale!(cu,Linv)
        LXZ = reshape(dl.Lb,(p,q))
        A_ldiv_B!(dl.Lt,BLAS.gemv!('N',-1.,LXZ,vec(cu),1.,β)) # solve for β
        BLAS.gemv!('T',-1.,LXZ,β,1.0,vec(cu)) # cu -= LZX'β
        scale!(cu,Linv)
    else
        for j in 1:size(cu,2)           # solve L cᵤ = λ'Z'y and downdate X'y
            BLAS.gemv!('T',-1.0,sub(dl.Lb,:,:,j),
                       BLAS.trsv!('L','N','N',sub(dl.Ld,:,:,j),sub(cu,:,j)),1.0,β)
        end
        A_ldiv_B!(dl.Lt,β)              # solve for β
        for j in 1:size(cu,2)           # solve L'u = cᵤ - R_ZX β
            BLAS.trsv!('L','T','N',sub(dl.Ld,:,:,j),
                       BLAS.gemv!('N',-1.0,sub(dl.Lb,:,:,j),β,1.0,sub(cu,:,j)))
        end
    end
    lmb
end

type DeltaNode <: Delta
    Ad::Vector{Delta}
    Ab::Array{Float64,3}
    At::Symmetric{Float64}
    Lb::Array{Float64,3}
    Lt::Base.LinAlg.Cholesky{Float64}
end
