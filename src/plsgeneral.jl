type PLSGeneral{Ti<:Union(Int32,Int64)} <: PLSSolver
    L::CholmodFactor{Float64,Ti}
    RX::Base.LinAlg.Cholesky{Float64}
    RZX::Matrix{Float64}
    XtX::Symmetric{Float64}
    Ztnz::Matrix{Float64}               # non-zeros in Zt as a dense matrix
    ZtX::Matrix{Float64}
    nlev::Vector
    perm::Vector{Ti}
    λtZt::CholmodSparse{Float64,Ti}
end

function PLSGeneral(Zt::SparseMatrixCSC,X::Matrix,facs::Vector)
    XtX = Symmetric(X'X,:L)
    ZtX = Zt*X
    Ztc = CholmodSparse!(Zt,0)
    cp = Ztc.colptr0
    d2 = cp[2] - cp[1]
    for j in 3:length(cp)
        cp[j] - cp[j-1] == d2 || error("Zt must have constant column counts")
    end
    L = cholfact(Ztc,1.,true)
    PLSGeneral(L,cholfact(XtX.S,:L),copy(ZtX),XtX,reshape(copy(Zt.nzval),(d2,Zt.n)),ZtX,
               [length(f.pool) for f in facs],L.Perm .+ one(eltype(L.Perm)),
               Ztc)
end

function Base.A_ldiv_B!(s::PLSGeneral,λ::Vector,Zty::Vector,Xty,u::Vector,β)
    cu = solve(s.L,permute!(vcat(map((x,y)-> vec(x*y),λ,Zty)...),s.perm),CHOLMOD_L)
    A_ldiv_B!(s.RX,BLAS.gemv!('T',-1.,s.RZX,cu,1.,copy!(β,Xty)))
    u = ipermute!(solve(s.L,BLAS.gemv!('N',-1.,s.RZX,β,1.,cu),CHOLMOD_Lt),s.perm)
    pos = 0
    for ui in u, j in 1:length(ui)
        ui[j] = u[pos += 1]
    end
    β
end

function update!(s::PLSGeneral,λ::Vector)
    λtZtm = reshape(copy!(s.λtZt.nzval,s.Ztnz),size(s.Ztnz))
    copy!(s.RZX,s.ZtX)
    Ztrow = 0
    ZtXrow = 0
    for k in 1:length(λ)
        ll = λ[k]
        p = size(ll,1)
        Ac_mul_B!(ll,sub(λtZtm,Ztrow + (1:p),:))
        Ztrow += p
        for i in 1:s.nlev[k]
            Ac_mul_B!(ll,sub(s.RZX,ZtXrow + (1:p),:))
            ZtXrow += p
        end
    end
    cholfact!(s.L,s.λtZt,1.)
    ## CHOLMOD doesn't solve in place so need to solve then copy
    copy!(s.RZX,solve(s.L, s.RZX[s.perm,:], CHOLMOD_L))
    _,info = LAPACK.potrf!('L',BLAS.syrk!('L','T',-1.,s.RZX,
                                          1.,copy!(s.RX.UL,s.XtX.S)))
    info == 0 || error("Downdated X'X is not positive definite")
    s
end
