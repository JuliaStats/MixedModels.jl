## A PLSSolver must implement methods for logdet, updateL! and A_ldiv_B! where B is LMMBase
type λtZtSolver{Ti<:Union(Int32,Int64)} <: PLSSolver # Full sparse Cholesky updating from λtZt
    L::CholmodFactor{Float64,Ti}
    λtZt::SparseMatrixCSC{Float64,Ti}
    RX::Base.LinAlg.Cholesky{Float64}
    perm::Vector{Ti}
    XtX::Symmetric{Float64}
end

##  cholfact(x, RX=true) -> the Cholesky factor of the downdated X'X or LambdatZt
Base.cholfact(s::λtZtSolver,RX=true) = RX ? s.RX : s.L

## Logarithm of the determinant of the generator matrix for the Cholesky factor, L or RX
Base.logdet(s::λtZtSolver,RX=true) = logdet(cholfact(s,RX))

function Base.A_ldiv_B!(lmb::LMMBase,s::λtZtSolver,ubeta::Bool=false)
    n,p,q,k = size(lmb)
    LamtZt = λtZt(lmb)
    if ubeta
        cu = solve(s.L,permute!(LamtZt*lmb.y,s.perm),CHOLMOD_L)
        RZX = LamtZt * lmb.X.m
        for j in 1:size(RZX,2)
            permute!(sub(RZX,:,j),s.perm)
        end
        RZX = solve(s.L, RZX, CHOLMOD_L)
        _,info = LAPACK.potrf!('U',BLAS.syrk!('U','T',-1.,RZX,1.,copy!(s.RX.UL,s.XtX.S)))
        info == 0 || error("downdated X'X is singular")
        LAPACK.potrs!('U',s.RX.UL,BLAS.gemv!('T',-1.,RZX,cu,1.,copy!(lmb.β,lmb.Xty)))
        u = ipermute!(solve(s.L,BLAS.gemv!('N',-1.,RZX,lmb.β,1.,cu),CHOLMOD_Lt),s.perm)
    else
        u = vec(solve(s.L,LamtZt * BLAS.gemv!('N',-1.0,lmb.X.m,lmb.β,1.0,copy(lmb.y))).mat)
    end
    pos = 0
    for u in lmb.u
        ll = length(u)
        u[:] = sub(u,pos+(1:ll))
        pos += ll
    end
end

## called after a θ! update on lmb
function updateL!(s::λtZtSolver,lmb::LMMBase)
    ## Need the version of λtZt that updates
end
