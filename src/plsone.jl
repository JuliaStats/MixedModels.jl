## Default methods, overridden for PLSOne

Base.cholfact(s::PLSSolver,RX::Bool=true) = RX ? s.RX : s.L
Base.logdet(s::PLSSolver,RX::Bool=true) = logdet(cholfact(s,RX))

type PLSOne <: PLSSolver   # Solver for models with a single random-effects term
    Ad::Array{Float64,3}                # diagonal blocks
    Ab::Array{Float64,3}                # base blocks
    At::Symmetric{Float64}              # lower right block
    Ld::Array{Float64,3}                # diagonal blocks
    Lb::Array{Float64,3}                # base blocks
    Lt::Base.LinAlg.Cholesky{Float64}   # lower right triangle
end

function PLSOne(Ad::Array{Float64,3}, Ab::Array{Float64,3}, At::Symmetric{Float64})
    m,n,t = size(Ad)
    m == n || error("Faces of Ad must be square")
    p,q,r = size(Ab)
    p == size(At,1) && q == n && r == t || error("Size mismatch")
    PLSOne(Ad,Ab,At,zeros(Ad),zeros(Ab),cholfact(At.S,symbol(At.uplo)))
end

function PLSOne(Ad::Array{Float64,3}, Ab::Array{Float64,3}, At::Matrix{Float64})
    PLSOne(Ad,Ab,Symmetric(At,:L))
end

function PLSOne(ff::PooledDataVector, Xst::Matrix, Xt::Matrix)
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
        BLAS.ger!(1.0,sub(Xt,:,j),sub(Xst,:,j),sub(Ab,:,:,jj))
    end
    for j in 1:nl        # symmetrize the faces created with BLAS.syr!
        Base.LinAlg.copytri!(sub(Ad,:,:,j),'L')
    end
    PLSOne(Ad,Ab,Symmetric(Xt*Xt',:L))
end

Base.cholfact(s::PLSOne,RX::Bool=true) = RX ? s.Lt : blkdiag({sparse(tril(sub(s.Ld,:,:,j))) for j in 1:size(s.Ld,3)}...)

## Logarithm of the determinant of the matrix represented by RX or L
function Base.logdet(s::PLSOne,RX=true)
    RX && return logdet(s.Lt)
    Ld = s.Ld
    m,n,t = size(Ld)
    s = 0.
    @inbounds for j in 1:t, i in 1:m
        s += log(Ld[i,i,j])
    end
    2.s
end

Base.size(s::PLSOne) = size(s.Ab)
Base.size(s::PLSOne,k::Integer) = size(s.Ab,k)

## function Base.Triangular{T<:Number}(n::Integer,v::Vector{T})
##     length(v) == n*(n+1)>>1 || error("Dimension mismatch")
##     A = zeros(T,n,n)
##     pos = 0
##     for j in 1:n, i in j:n
##         A[i,j] = v[pos += 1]
##     end
##     Triangular(A,:L,false)
## end

##  update!(s,lambda)->s : update Ld, Lb and Lt
function update!(s::PLSOne,λ::Triangular)
    m,n,l = size(s)
    n == size(λ,1) || error("Dimension mismatch")
    Lt = copy!(s.Lt.UL,s.At.S)
    Lb = copy!(s.Lb,s.Ab)
    if n == 1                           # shortcut for 1×1 λ
        lam = λ[1,1]
        Ld = map!(x -> sqrt(x*lam*lam + 1.), s.Ld, s.Ad)
        Lb = scale!(reshape(Lb,(m,n*l)),lam ./ vec(Ld))
        BLAS.syrk!('L','N',-1.0,Lb,1.0,Lt)
    else
        Ac_mul_B!(λ,reshape(copy!(s.Ld,s.Ad),(n,n*l)))
        for k in 1:l
            wL = A_mul_B!(sub(s.Ld,:,:,k),λ)
            for j in 1:n                # Inflate the diagonal
                wL[j,j] += 1.
            end
            _, info = LAPACK.potrf!('L',wL) # i'th diagonal block of L
            info == 0 || error("Cholesky failure at L diagonal block $k")
            Base.LinAlg.A_rdiv_Bc!(A_mul_B!(sub(s.Lb,:,:,k),λ),Triangular(wL,:L,false))
        end
        BLAS.syrk!('L','N',-1.0,reshape(Lb,(m,n*l)),1.,Lt)
    end
    _, info = LAPACK.potrf!('L',Lt)
    info == 0 ||  error("downdated X'X is not positive definite")
    s
end

function update!(s::PLSOne,λ::Vector)
    length(λ) == 1 || error("update! on a PLSOne requires length(λ) == 1")
    update!(s,λ[1])
end

## arguments passed contain λ'Z'y and X'y
function Base.A_ldiv_B!(s::PLSOne,u::Vector,β)
    length(u) == 1 || error("length(u) = $(length(u)), should be 1 for PLSOne")
    p,k,l = size(s)
    cu = u[1]
    (q = length(cu)) == k*l && k == size(cu,1) || error("Dimension mismatch")
    if k == 1                           # short cut for scalar r.e.
        Linv = 1. ./ vec(s.Ld)
        scale!(cu,Linv)
        LXZ = reshape(s.Lb,(p,k*l))
        A_ldiv_B!(s.Lt,BLAS.gemv!('N',-1.,LXZ,vec(cu),1.,β)) # solve for β
        BLAS.gemv!('T',-1.,LXZ,β,1.0,vec(cu)) # cu -= LZX'β
        scale!(cu,Linv)
    else
        for j in 1:l                    # solve L cᵤ = λ'Z'y blockwise
            BLAS.trsv!('L','N','N',sub(s.Ld,:,:,j),sub(cu,:,j))
        end
                                        # solve (L_X L_X')̱β = X'y - L_XZ cᵤ
        A_ldiv_B!(s.Lt,BLAS.gemv!('N',-1.0,reshape(s.Lb,(p,q)),vec(cu),1.0,β))
                                        # cᵤ := cᵤ - L_XZ'β
        BLAS.gemv!('T',-1.0,reshape(s.Lb,(p,q)),β,1.0,vec(cu))
        for j in 1:l                    # solve L'u = cᵤ blockwise
            BLAS.trsv!('L','T','N',sub(s.Ld,:,:,j),sub(cu,:,j))
        end
    end
end

function grad(s::PLSOne,facs::Vector,sc,u::Vector,Xs::Vector,Zty::Vector,λ::Vector,μ)
    length(u) == 1 || error("PLSOne must correspond to a LMMBase with 1 r.e. term")
    u = u[1]
    λ = λ[1]
    nz = Xs[1]
    res = zeros(size(λ))
    rv = facs[1].refs
    tmp = similar(res)                  # scratch array
    for i in 1:size(s.Ad,3)
        res += LAPACK.potrs!('L',sub(s.Ld,:,:,i),Ac_mul_B!(λ,copy!(tmp,sub(s.Ad,:,:,i))))
    end
    Ztr = copy(Zty[1])          # create Z'(resid) starting with Zty
    for i in 1:length(μ)
        Ztr[:,rv[i]] -= μ[i] * nz[:,i]
    end
    ltri(BLAS.syr2k!('L','N',-1./sc,Ztr,u,1.,res+res'))
end
