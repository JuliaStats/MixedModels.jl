type PLSDiagWSMP <: PLSSolver # Sparse Choleksy solver for diagonal Λ
    LX::Base.LinAlg.Cholesky{Float64}
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
    PLSDiagWSMP(cholfact(XtX.S,:L),similar(ZtX),W,XtX,ZtX,copy(W.avals),copy(W.diag),
                cumsum([length(ff.pool)::Int for ff in facs]),ones(size(Zt,1)))
end

function plssolve!(s::PLSDiagWSMP,u::Vector,β)
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
    lam = λ[1].s.λ
    for j in 1:length(s.λvec)
        j > s.trmsz[ind] && (lam = λ[ind += 1].s.λ)
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
    _,info = LAPACK.potrf!('L',BLAS.syrk!('L','T',-1.,s.RZX,1.,copy!(s.LX.UL,s.XtX.S)))
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

type PLSDiagWA <: PLSSolver
    W::Wssmp
    avals::Vector{Float64}
    diag::Vector{Float64}
    trmsz::Vector{Int}
    λvec::Vector{Float64}
end
