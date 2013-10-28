## Convert a random-effects term t to a model matrix, a factor and a name
function retrm(t::Expr,df::DataFrame)
    grp = t.args[3]
    X = ones(nrow(df),1)
    if t.args[2] != 1
        template = Formula(:(~ foo)); template.rhs=t.args[2]
        X = ModelMatrix(ModelFrame(template, df)).m
    end
    X, pool(df[grp]), string(grp)
end

function lmm(f::Formula, fr::AbstractDataFrame, forcegeneral::Bool=false)
    mf = ModelFrame(f, fr); X = ModelMatrix(mf); y = model_response(mf)
    reinfo = {retrm(r,mf.df) for r in
        filter(x->Meta.isexpr(x,:call) && x.args[1] == :|, mf.terms.terms)}
    (k = length(reinfo)) > 0 || error("Formula $f has no random-effects terms")
    if k == 1 && !forcegeneral
        Xs, fac, nm = reinfo[1]
        size(Xs,2) == 1 && return LMMScalar1(X,Xs,fac,y,nm) 
        return LMMVector1(X,Xs,fac,y,nm)
    end
    Xs = Matrix{Float64}[r[1] for r in reinfo]; facs = [r[2] for r in reinfo]
    fnms = String[r[3] for r in reinfo]; pvec = Int[size(m,2) for m in Xs]
    refs = [f.refs for f in facs]; levs = [f.pool for f in facs]; n,p = size(X)
    nlev = [length(l) for l in levs]; offsets = [0, cumsum(nlev)]
    !forcegeneral && all([isnested(refs[i-1],refs[i]) for i in 2:k]) &&
        return LMMNested(X,Xs,refs,levs,y,fnms,pvec,nlev,offsets)
    ## Other cases use CHOLMOD code with index type in Union(Int32,Int64)
    q = sum(nlev .* pvec); Ti = q < typemax(Int32) ? Int32 : Int64
    Zt = SparseMatrixCSC(offsets[end],n,convert(Vector{Ti},[1:k:(k*n + 1)]),
                         convert(Vector{Ti},vec(broadcast(+,hcat(refs...)',
                                                          offsets[1:k]))),
                         vec(hcat(Xs...)'))
    Ztc = CholmodSparse(Zt)
    ZtZ = Ztc * Ztc'; L = cholfact(ZtZ,1.,true); perm = L.Perm + one(Ti)
    !forcegeneral && all(pvec .== 1) &&
        return LMMScalarn{Ti}(copy(ZtZ),L,Diagonal(ones(q)),cholfact(eye(p)),
                              X,X.m'*y,Zt,Zt*X.m,ZtZ,Zt*y,zeros(p),fnms,
                              ones(k),zeros(n),offsets,perm,
                              Vector{Float64}[ones(j) for j in nlev],
                              y,false,false)
    LMMGeneral(L,Ztc,cholfact(eye(p)),X,Xs,
               Symmetric(syrk!('U','T',1.,X.m,0.,zeros(p,p)),:U),
               X.m'*y,zeros(p),refs,Matrix{Float64}[eye(p) for p in pvec],
               zeros(n),perm,pvec,
               Matrix{Float64}[zeros(pvec[i],nlev[i]) for i in 1:k],
               y,false,false)
end

lmm(ex::Expr, fr::AbstractDataFrame) = lmm(Formula(ex), fr)
