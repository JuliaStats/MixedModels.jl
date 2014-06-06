type GenSolver{Ti<:Union(Int32,Int64)} <: PLSSolver
    L::CholmodFactor{Float64,Ti}
    RX::Base.LinAlg.Cholesky{Float64}
    RZX::Matrix{Float64}
    XtX::Symmetric{Float64}
    Zt::SparseMatrixCSC{Float64,Ti}
    ZtX::Matrix{Float64}
    ZtZ::CholmodSparse{Float64,Ti}
    perm::Vector{Ti}
    nlev::Vector
end

function GenSolver(lmb::LMMBase)        # incomplete
    Zt = zt(lmb)
    Ztc = CholmodSparse(Zt)
    ZtZ = Ztc * Ztc'
    L = cholfact(ZtZ,1.,true)
    perm = L.Perm .+ one(eltype(L.Perm))
    X = lmb.X.m
    XtX = Symmetric(X'X,:L)
    ZtX = Zt*X
    GenSolver(L,cholfact(XtX.S,:L),copy(ZtX),XtX,Zt,ZtX,ZtZ,
              perm,[length(f.pool) for f in lmb.facs])
end

function Base.A_ldiv_B!(s::GenSolver,lmb::LMMBase)
    cu = solve(s.L,permute!(vcat(map((x,y)-> vec(x*y),lmb.λ,lmb.Zty)...),s.perm),CHOLMOD_L)
    A_ldiv_B!(s.RX,BLAS.gemv!('T',-1.,s.RZX,cu,1.,copy!(lmb.β,lmb.Xty)))
    u = ipermute!(solve(s.L,BLAS.gemv!('N',-1.,s.RZX,lmb.β,1.,cu),CHOLMOD_Lt),s.perm)
    pos = 0
    for ui in lmb.u, j in 1:length(ui)
        ui[j] = u[pos += 1]
    end
    lmb
end

function update!(s::GenSolver,λ::Vector)
    Λ = convert(typeof(s.Zt),blkdiag({kron(full(l),speye(nl)) for (l,nl) in zip(λ,s.nlev)}...))
    cholfact!(s.L,Λ*s.Zt,1.)
    copy!(s.RZX,solve(s.L, (Λ * s.ZtX)[s.perm,:], CHOLMOD_L))
    _,info = LAPACK.potrf!('L',BLAS.syrk!('L','T',-1.,s.RZX,1.,copy!(s.RX.UL,s.XtX.S)))
    info == 0 || error("Downdated X'X is not positive definite")
    s
end

## Try storing the Λ matrix and an Lind vector instead.  Less overhead of copying.
