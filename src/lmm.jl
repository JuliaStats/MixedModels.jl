const template = Formula(:(~ foo))      # for evaluating the lhs of r.e. terms

## Convert a random-effects term t to a model matrix, a factor and a name
function retrm(t::Expr,df::DataFrame)
    X = t.args[2] == 1 ? ones(nrow(df),1) :
           (template.rhs=t.args[2]; ModelMatrix(ModelFrame(template, df)).m)
    grp = t.args[3]
    X, PooledDataArray(df[grp]), string(grp)
end

function lmm(f::Formula, fr::AbstractDataFrame)
    mf = ModelFrame(f, fr); X = ModelMatrix(mf); y = model_response(mf)
    reinfo = [retrm(r,mf.df) for r in
              filter(x->Meta.isexpr(x,:call) && x.args[1] == :|, mf.terms.terms)]
    (k = length(reinfo)) > 0 || error("Formula $f has no random-effects terms")
    if k == 1
        Xs, fac, nm = reinfo[1]
        size(Xs,2) == 1 && return LMMScalar1(X,Xs,fac,y,nm) 
        return LMMVector1(X,Xs,fac,y,nm)
    end
    Xs = Matrix{Float64}[r[1] for r in reinfo]; facs = [r[2] for r in reinfo]
    fnms = String[r[3] for r in reinfo]
    all((pvec = Int[size(m,2) for m in Xs]).==1) && return Scalarn(X,Xs,facs,y,fnms)
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
