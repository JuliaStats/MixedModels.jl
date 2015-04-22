type PLSDiagWSMP <: PLSSolver # Sparse Choleksy solver for diagonal Λ
    LX::Base.Cholesky{Float64}
    RZX::Matrix{Float64}
    W::Wssmp
    XtX::Symmetric{Float64}
    ZtX::Matrix{Float64}
    avals::Vector{Float64}
    diag::Vector{Float64}
    trmsz::Vector{Int}
    λvec::Vector{Float64}
end

function PLSDiagWSMP(Zt::SparseMatrixCSC,X::Matrix,facs::Vector)
    W = Wssmp(Zt * Zt',true)
    wssmp(W,2)      # ordering and symbolic factorization
    W.iparm[32] = 1 # overwrite W.diag by diagonal of L in numeric step
    XtX = Symmetric(X'X,:L)
    ZtX = Zt*X
    PLSDiagWSMP(cholfact(XtX.data,:L),similar(ZtX),W,XtX,ZtX,copy(W.avals),copy(W.diag),
                cumsum([length(ff.pool)::Int for ff in facs]),ones(size(Zt,1)))
end

function Base.A_ldiv_B!(s::PLSDiagWSMP,u::Vector,β)
    s.W.iparm[30] = 1
    wssmp(s.W,u)
    A_ldiv_B!(s.LX,BLAS.gemv!('T',-1.,s.RZX,u,1.,β))
    s.W.iparm[30] = 2
    wssmp(s.W,BLAS.gemv!('N',-1.,s.RZX,β,1.,u))
end
    
function update!(s::PLSDiagWSMP,λ::Vector)
    for ll in λ
        isa(ll,PDScalF) || error("λ must be a vector PDScalF objects")
    end
    ind = 1
    lam = λ[1].s
    for j in 1:length(s.λvec)
        j > s.trmsz[ind] && (lam = λ[ind += 1].s)
        s.λvec[j] = lam
    end
    for (j,lam) in enumerate(s.λvec)
        s.W.diag[j] = abs2(lam) * s.diag[j] + 1. # scale and inflate diagonal
        for k in s.W.ia[j]:(s.W.ia[j] - 1)
            s.W.avals[k] = lam * s.avals[k] * s.λvec[s.ja[k]]
        end
    end
    scale!(s.RZX,s.λvec,s.ZtX)
    s.W.iparm[2] = 3
    s.W.iparm[30] = 1
    wssmp(s.W,s.RZX,4)
    _,info = LAPACK.potrf!('L',BLAS.syrk!('L','T',-1.,s.RZX,1.,copy!(chfac(s.LX),s.XtX.data)))
    info == 0 || error("Downdated X'X is not positive definite")
    s
end

function Base.logdet(s::PLSDiagWSMP,RX::Bool=true)
    RX && return logdet(s.LX)
    ld = 0.
    for d in s.W.diag
        ld += log(d)
    end
    2.ld
end

type PLSDiagWA <: PLSSolver             # PLS solver using WSMP for both random and fixed
    W::Wssmp
    avals::Vector{Float64}
    trmsz::Vector{Int}
    λvec::Vector{Float64}
end

function PLSDiagWA(Zt::SparseMatrixCSC{Cdouble,Cint},X::Matrix{Float64},facs::Vector)
    ## ppq = size(Zt,1) + size(X,2)
    ## perm = Int32[1:ppq]
    ## invp = copy(perm)
    ## adj = Base.SparseMatrix.fkeep!(Zt*Zt',(i,j,x,other) -> (i≠j), None)
    ## ccall((:wkktord_,WSMP.libwsmp),Void,(Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint},
    ##                                      Ptr{Cint},Ptr{Cint},Ptr{Cint},Ptr{Cint}),
    ##       &size(adj,1),adj.colptr,adj.rowval,Int32[3,0,1,0,0],&1,perm,invp,C_NULL,&0)
    ZXt = vcat(Zt,convert(typeof(Zt),X'))
    W = Wssmp(ZXt*ZXt')
    W.iparm[32] = 1        # store the diagonal of the Cholesky factor
    W.diag = Array(Cdouble,size(ZXt,1))
    W.iparm[15] = size(Zt,1) # restrict ordering not to mix X and Z parts
    wssmp(W,2)
    PLSDiagWA(W,copy(W.avals),[length(f.pool) for f in facs],ones(size(ZXt,1)))
end

Base.A_ldiv_B!(s::PLSDiagWA,uβ::Vector) = A_ldiv_B!(s.W,uβ)

function Base.logdet(s::PLSDiagWA,RX::Bool=true)
    q = sum(s.trmsz)
    dd = s.W.diag
    sm = 0.
    for i in (RX ? ((q+1):length(s.λvec)) : (1:q))
        sm += log(dd[i])
    end
    2.sm
end

function update!(s::PLSDiagWA,λ::Vector)
    length(λ) == length(s.trmsz) || throw(DimensionMismatch(""))
    for ll in λ
        isa(ll,PDScalF) || error("λ must be a vector PDScalF objects")
    end
    λvec = s.λvec
    offset = 0
    for (ll,sz) in zip(λ,s.trmsz)
        fill!(ContiguousView(λvec,offset,(sz,)),ll.s.λ)
        offset += sz
    end
    q = sum(s.trmsz)
    colptr = s.W.ia
    rowval = s.W.ja
    nzval = s.W.avals
    for j in 1:length(λvec), k in colptr[j]:(colptr[j+1]-1)
        i = rowval[k]
        nzval[k] = s.avals[k] * λvec[j] * λvec[i] # scale rows and columns
        i == j && i ≤ q && (nzval[k] += 1.) # inflate the Z part's diagonal
    end
    s.W.iparm[2] = 3
    wssmp(s.W,3)
    s
end
