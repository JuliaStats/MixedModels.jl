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

function lmm(f::Formula, fr::AbstractDataFrame)
    mf = ModelFrame(f, fr); X = ModelMatrix(mf); y = model_response(mf)
    rtrms = filter(x->Meta.isexpr(x,:call) && x.args[1] == :|, mf.terms.terms)
    reinfo = [retrm(r,mf.df) for r in rtrms]
    ## reinfo = [retrm(r,mf.df) for r in
    ##           filter(x->Meta.isexpr(x,:call) && x.args[1] == :|, mf.terms.terms)]
    (k = length(reinfo)) > 0 || error("Formula $f has no random-effects terms")
    if k == 1
        Xs, fac, nm = reinfo[1]
        size(Xs,2) == 1 && return LMMScalar1(X,Xs,fac,y,nm) 
        return LMMVector1(X,Xs,fac,y,nm)
    end
    Xs = Matrix{Float64}[r[1] for r in reinfo]; facs = [r[2] for r in reinfo]
    fnms = String[r[3] for r in reinfo]; pvec = Int[size(m,2) for m in Xs]
    refs = [f.refs for f in facs]; levs = [f.pool for f in facs]; n,p = size(X)
    nlev = [length(l) for l in levs]; offsets = [0, cumsum(nlev)]
    all([isnested(refs[i-1],refs[i]) for i in 2:k]) &&
        return LMMNested(X,Xs,refs,levs,y,fnms,pvec,nlev,offsets)
    q = sum(nlev .* pvec); Ti = q < typemax(Int32) ? Int32 : Int64
    Zt = SparseMatrixCSC(offsets[end],n,convert(Vector{Ti},[1:k:(k*n + 1)]),
                         convert(Vector{Ti},vec(broadcast(+,hcat(refs...)',
                                                          offsets[1:k]))),
                         vec(hcat([x[:,1] for x in Xs]...)'))
    Ztc = CholmodSparse(Zt)
    ZtZ = Ztc * Ztc'; L = cholfact(ZtZ,1.,true); perm = L.Perm + one(Ti)
    all(pvec .== 1) &&
        return LMMScalarn{Ti}(copy(ZtZ),L,Diagonal(ones(q)),cholfact(eye(p)),
                              X,X.m'*y,Zt,Zt*X.m,ZtZ,Zt*y,zeros(p),fnms,
                              ones(k),zeros(n),offsets,perm,
                              Vector{Float64}[ones(j) for j in nlev],
                              y,false,false)
    error("should not reach here")
    LMMGeneral(X,Xs,facs,y,fnms,pvec)
end
        
## lambda = similar(u); inds = Array(Any,k); rowval = Array(Matrix{Int},k)
##     offsets = zeros(Int, k + 1); nlev = zeros(Int,k); ncol = zeros(Int,k)
##     for i in 1:k
##         t = re[i]                           # i'th random-effects term
##         gf = PooledDataArray(df[t.args[3]]) # i'th grouping factor
##         nlev[i] = l = length(gf.pool); inds[i] = gf.refs; 
##         if t.args[2] == 1                   # simple, scalar term
##             Xs[i] = ones(n,1); ncol[i] = p = 1; lambda[i] = ones(1,1)
##         else
##             Xs[i] = (template.rhs=t.args[2]; ModelMatrix(ModelFrame(template, df))).m
##             ncol[i] = p = size(Xs[i],2); lambda[i] = eye(p)
##         end
##         p <= 1 || (scalar = false)
##         nu = p * l; offsets[i + 1] = offsets[i] + nu;
##         rowval[i] = (reshape(1:nu,(p,l)) + offsets[i])[:,inds[i]]
##     end
##     q = offsets[end]; uvec = Array(Float64,q);
##     Ti = Int; if q < typemax(Int32); Ti = Int32; end # use 32-bit ints if possible
##     rv = convert(Matrix{Ti},vcat(rowval...))
##     for i in 1:k                                     # view uvec as a vector of matrices
##         u[i] = pointer_to_array(convert(Ptr{Float64},sub(uvec,(offsets[i]+1):offsets[i+1])),
##                                 (ncol[i],nlev[i]))
##     end
##                                      # create the appropriate type of LMM object
##     scalar && k == 1 && return LMMScalar1(X.m',vec(rv),vec(Xs[1]),y)
##     scalar && return LMMScalarn(q,X,Xs,u,rv,y)
##     k == 1 && return LMMVector1(q,X,Xs,u,rv,y)
##     LMMGeneral(q,X,Xs,inds,y,rv,y,lambda)
## end
lmm(ex::Expr, fr::AbstractDataFrame) = lmm(Formula(ex), fr)
