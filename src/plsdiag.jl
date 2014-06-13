type PLSDiag{Ti<:Union(Int32,Int64)} <: PLSSolver # Sparse Choleksy solver with diagonal Λ
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
    perm = L.Perm
    XtX = Symmetric(X'X,:L)
    ZtX = Zt*X
    PLSDiag(L,cholfact(XtX.S,:L),copy(ZtX),XtX,ZtX,ZtZ,perm .+ one(eltype(perm)),
               vcat([fill(j,length(ff.pool)) for (j,ff) in enumerate(facs)]...))
end

function Base.A_ldiv_B!(s::PLSDiag,u::Vector,β)
    cu = solve(s.L,permute!(vec(hcat(u...)),s.perm),CHOLMOD_L)
    A_ldiv_B!(s.RX,BLAS.gemv!('T',-1.,s.RZX,cu,1.,β))
    ipermute!(solve(s.L,BLAS.gemv!('N',-1.,s.RZX,β,1.,cu),CHOLMOD_Lt),s.perm)
    pos = 0
    for ui in u, j in 1:length(ui)
        ui[j] = cu[pos += 1]
    end
end

function update!(s::PLSDiag,λ::Vector)
    all(map(size,λ) .== (1,1)) || error("λ must be a vector of 1×1 matrices")
    λvec = vcat([vec(ll.data) for ll in λ]...)[s.λind]
    cholfact!(s.L,chm_scale(s.ZtZ,λvec,CHOLMOD_SYM),1.)
    copy!(s.RZX,solve(s.L, scale(λvec, s.ZtX)[s.perm,:], CHOLMOD_L))
    _,info = LAPACK.potrf!('L',BLAS.syrk!('L','T',-1.,s.RZX,1.,copy!(s.RX.UL,s.XtX.S)))
    info == 0 || error("Downdated X'X is not positive definite")
    s
end
