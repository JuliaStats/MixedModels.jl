function lmm(f::Formula, fr::AbstractDataFrame)
    lmb = LMMBase(f,fr)
    n,p,q,k = size(lmb)
    isscalar(lmb) && return k == 1 ? LMMScalar1(lmb) : LMMScalarn(lmb)
    k == 1 ? LMMVector1(lmb) : LMMGeneral(lmb)
end
    
lmm(ex::Expr, fr::AbstractDataFrame) = lmm(Formula(ex), fr)
