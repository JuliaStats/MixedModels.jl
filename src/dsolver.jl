type DiagSolver{Ti<:Union(Int32,Int64)} <: PLSSolver # Sparse Choleksy solver with diagonal Λ
    L::CholmodFactor{Float64,Ti}
    RX::Base.LinAlg.Cholesky{Float64}
    RZX::Matrix{Float64}
    XtX::Symmetric{Float64}
    ZtX::Matrix{Float64}
    ZtZ::CholmodSparse{Float64,Ti}
    perm::Vector{Ti}
    λind::Vector
end

function DiagSolver(lmb::LMMBase)
    zt = Zt(lmb)
    Ztc = CholmodSparse(zt)
    ZtZ = Ztc * Ztc'
    L = cholfact(ZtZ,1.,true)
    perm = L.Perm
    X = lmb.X.m
    XtX = Symmetric(X'X,:L)
    ZtX = zt*X
    DiagSolver(L,cholfact(XtX.S,:L),copy(ZtX),XtX,ZtX,ZtZ,perm .+ one(eltype(perm)),
               vcat([fill(j,length(ff.pool)) for (j,ff) in enumerate(lmb.facs)]...))
end

function Base.A_ldiv_B!(s::DiagSolver,lmb::LMMBase)
    cu = solve(s.L,permute!(vec(hcat(map(*,lmb.λ,lmb.Zty)...)),s.perm),CHOLMOD_L)
    A_ldiv_B!(s.RX,BLAS.gemv!('T',-1.,s.RZX,cu,1.,copy!(lmb.β,lmb.Xty)))
    u = ipermute!(solve(s.L,BLAS.gemv!('N',-1.,s.RZX,lmb.β,1.,cu),CHOLMOD_Lt),s.perm)
    pos = 0
    for ui in lmb.u, j in 1:length(ui)
        ui[j] = u[pos += 1]
    end
    lmb
end

Base.cholfact(s::DiagSolver,RX::Bool=true) = RX ? s.RX : s.L

Base.logdet(s::DiagSolver,RX::Bool=true) = logdet(cholfact(s,RX))

function update!(s::DiagSolver,λ::Vector)
    all(map(size,λ) .== (1,1)) || error("λ must be a vector of 1×1 matrices")
    λvec = vcat([vec(ll.UL) for ll in λ]...)[s.λind]
    cholfact!(s.L,chm_scale(s.ZtZ,λvec,CHOLMOD_SYM),1.)
    copy!(s.RZX,solve(s.L, scale(λvec, s.ZtX)[s.perm,:], CHOLMOD_L))
    _,info = LAPACK.potrf!('L',BLAS.syrk!('L','T',-1.,s.RZX,1.,copy!(s.RX.UL,s.XtX.S)))
    s
end


