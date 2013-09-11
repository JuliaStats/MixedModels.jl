const template = Formula(:(~ foo))      # for evaluating the lhs of r.e. terms

function lmm(f::Formula, fr::AbstractDataFrame; dofit=true)
    mf = ModelFrame(f, fr); df = mf.df; n = size(df,1)
    X = ModelMatrix(mf); y = float64(vector(model_response(mf)))
                                        # extract and check random-effects terms
    re = filter(x->Meta.isexpr(x,:call) && x.args[1] == :|, mf.terms.terms)
    (k = length(re)) > 0 || error("Formula $f has no random-effects terms")
                                        # quantities derived from r.e. terms
    scalar = true; u = Array(Matrix{Float64},k); Xs = similar(u)
    lambda = similar(Xs); inds = Array(Any,k); rowval = Array(Matrix{Int},k)
    offsets = zeros(Int, k + 1); nlev = zeros(Int,k); ncol = zeros(Int,k)
    for i in 1:k
        t = re[i]                           # i'th random-effects term
        gf = PooledDataArray(df[t.args[3]]) # i'th grouping factor
        nlev[i] = l = length(gf.pool); inds[i] = gf.refs; 
        if t.args[2] == 1                   # simple, scalar term
            Xs[i] = ones(n,1); ncol[i] = p = 1; lambda[i] = ones(1,1)
        else
            Xs[i] = (template.rhs=t.args[2]; ModelMatrix(ModelFrame(template, df))).m
            ncol[i] = p = size(Xs[i],2); lambda[i] = eye(p)
        end
        p <= 1 || (scalar = false)
        nu = p * l; offsets[i + 1] = offsets[i] + nu;
        rowval[i] = (reshape(1:nu,(p,l)) + offsets[i])[:,inds[i]]
    end
    q = offsets[end]; uvec = Array(Float64,q);
    Ti = Int; if q < typemax(Int32); Ti = Int32; end # use 32-bit ints if possible
    rv = convert(Matrix{Ti},vcat(rowval...))
    for i in 1:k                                     # view uvec as a vector of matrices
        u[i] = pointer_to_array(convert(Ptr{Float64},sub(uvec,(offsets[i]+1):offsets[i+1])),
                                (ncol[i],nlev[i]))
    end
    local m
                                     # create the appropriate type of LMM object
    if k == 1 && scalar
        m = LMMScalar1(X.m', vec(rv), vec(Xs[1]), y)
    else
        p = size(X.m,2); nz = hcat(Xs...)'
        LambdatZt = CholmodSparse!(convert(Vector{Ti}, [1:size(nz,1):length(nz)+1]),
                                   vec(copy(rv)), vec(nz), q, n, 0)
        L = cholfact(LambdatZt,1.,true); pp = invperm(L.Perm + one(Ti))
        rowvalperm = Ti[pp[rv[i,j]] for i in 1:size(rv,1), j in 1:size(rv,2)]

        m = LMMGeneral{Ti}(L,LambdatZt,cholfact(eye(p)),X,Xs,zeros(p),inds,lambda,
            zeros(n),vec(rowvalperm),u,y,false,false)
    end
    dofit ? fit(m) : m
end
lmm(ex::Expr, fr::AbstractDataFrame) = lmm(Formula(ex), fr)
