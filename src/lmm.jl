function lmm(f::Formula, fr::AbstractDataFrame)
    lmb = LMMBase(f,fr)
    n,p,q,k = size(lmb)
    if isscalar(lmb)
        k == 1 && return LMMScalar1(lmb)
        isnested(lmb) && return LMMScalarNested(lmb)
        return LMMScalarn(lmb)
    end
    k == 1 ? LMMVector1(lmb) : LMMGeneral(lmb)
end
    
lmm(ex::Expr, fr::AbstractDataFrame) = lmm(Formula(ex), fr)
