type PLSDiag{Ti<:Union(Int32,Int64)} <: PLSSolver # Sparse Choleksy solver for diagonal Λ
    L::CholmodFactor{Float64,Ti}
    RX::Base.LinAlg.Cholesky{Float64}
    RZX::Matrix{Float64}
    XtX::Symmetric{Float64}
    ZtX::Matrix{Float64}
    ZtZ::CholmodSparse{Float64,Ti}
    perm::Vector{Ti}
    λind::Vector
end

function PLSDiag(Zt::SparseMatrixCSC,X::Matrix,facs::Vector)
    Ztc = CholmodSparse(Zt)
    ZtZ = Ztc * Ztc'
    L = cholfact(ZtZ,1.,true)
    XtX = Symmetric(X'X,:L)
    ZtX = Zt*X
    PLSDiag(L,cholfact(XtX.S,:L),copy(ZtX),XtX,ZtX,ZtZ,L.Perm .+ one(eltype(L.Perm)),
            vcat([fill(j,length(ff.pool)) for (j,ff) in enumerate(facs)]...))
end

function plssolve!(s::Union(PLSDiag,PLSGeneral),u::Vector,β)
    copy!(u,solve(s.L,permute!(u,s.perm),CHOLMOD_L))
    A_ldiv_B!(s.RX,BLAS.gemv!('T',-1.,s.RZX,u,1.,β))
    copy!(u,ipermute!(solve(s.L,BLAS.gemv!('N',-1.,s.RZX,β,1.,u),CHOLMOD_Lt),s.perm))
end

function update!(s::PLSDiag,λ::Vector)
    for ll in λ
        isa(ll,PDScalF) || error("λ must be a vector PDScalF objects")
    end
    λvec = [ll.s.λ::Float64 for ll in λ][s.λind]
    cholfact!(s.L,chm_scale(s.ZtZ,λvec,CHOLMOD_SYM),1.)
    copy!(s.RZX,solve(s.L, scale(λvec, s.ZtX)[s.perm,:], CHOLMOD_L))
    _,info = LAPACK.potrf!('L',BLAS.syrk!('L','T',-1.,s.RZX,1.,copy!(s.RX.UL,s.XtX.S)))
    info == 0 || error("Downdated X'X is not positive definite")
    s
end
