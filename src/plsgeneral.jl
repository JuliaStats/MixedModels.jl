type PLSGeneral{Ti<:Union{Int32,Int64}} <: PLSSolver
    L::CHMfac{Float64}
    RX::Base.Cholesky{Float64}
    RZX::Matrix{Float64}
    XtX::Symmetric{Float64}
    Ztnz::Matrix{Float64}               # non-zeros in Zt as a dense matrix
    ZtX::Matrix{Float64}
    cu::Vector{Float64}
    nlev::Vector
    perm::Vector{Ti}
    λtZt::CHMsp{Float64}
end

function PLSGeneral(Zt::SparseMatrixCSC,X::Matrix,facs::Vector)
    XtX = Symmetric(X'X,:L)
    ZtX = Zt*X
    cp = Zt.colptr
    d2 = cp[2] - cp[1]
    for j in 3:length(cp)
        cp[j] - cp[j-1] == d2 || error("Zt must have constant column counts")
    end
    Ztc = CHMsp(Zt,0)
    L = VERSION < v"0.4-" ? cholfact(Ztc,1.,true) : cholfact(Ztc; shift = 1.)
    PLSGeneral(L,cholfact(symcontents(XtX),:L),copy(ZtX),XtX,
               reshape(copy(Zt.nzval),(Int(d2),Zt.n)), ZtX,zeros(size(L,1)),
               [length(f.pool) for f in facs],
               L.Perm .+ one(eltype(L.Perm)), Ztc)
end

function update!(s::PLSGeneral,λ::Vector)
    λtZtm = reshape(copy!(s.λtZt.nzval,s.Ztnz),size(s.Ztnz))
    copy!(s.RZX,s.ZtX)
    Ztrow = 0
    ZtXrow = 0
    for k in 1:length(λ)
        ll = λ[k]
        isa(ll, AbstractPDMatFactor) || error("isa(λ[$k],AbstractPDMatFactor) fails")
        p = abs(dim(ll))
        Ac_mul_B!(ll,view(λtZtm,Ztrow + (1:p),:))
        Ztrow += p
        for i in 1:s.nlev[k]
            Ac_mul_B!(ll,view(s.RZX,ZtXrow + (1:p),:))
            ZtXrow += p
        end
    end
    VERSION < v"0.4-" ? cholfact!(s.L,s.λtZt,1.) : Base.SparseMatrix.CHOLMOD.update!(s.L, s.λtZt; shift = 1.0)
    ## CHOLMOD doesn't solve in place so need to solve then copy
    copy!(s.RZX,solve(s.L, s.RZX[s.perm,:], CHOLMOD_L))
    _,info = LAPACK.potrf!('L',BLAS.syrk!('L','T',-1.,s.RZX,
                                          1.,copy!(chfac(s.RX),symcontents(s.XtX))))
    info == 0 || error("Downdated X'X is not positive definite")
    s
end
