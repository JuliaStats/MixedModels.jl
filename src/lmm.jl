function lmm(f::Formula, fr::AbstractDataFrame)
    lmb = LMMBase(f,fr)
    n,p,q,k = size(lmb)
    k == 1 && return LinearMixedModel(lmb,DeltaLeaf(lmb))
    isscalar(lmb) && return LinearMixedModel(lmb,DiagSolver(lmb))
    LinearMixedModel(lmb,GeneralSolver(lmb))
end
    
lmm(ex::Expr, fr::AbstractDataFrame) = lmm(Formula(ex), fr)
