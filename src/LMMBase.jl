## Note that linpred! methods can be applied here.

## Information common to all LinearMixedModel types
## Many methods for LinearMixedModel pass through to this type
type LMMBase
    f::Formula
    mf::ModelFrame
    X::ModelMatrix{Float64}
    y::Vector{Float64}
    res::Vector{Float64}
    mu::Vector{Float64}
    fnms::Vector                        # names of grouping factors
    facs::Vector
    Xs::Vector
end

function LMMBase(f::Formula, fr::AbstractDataFrame)
    mf = ModelFrame(f,fr)
    y = convert(Vector{Float64},model_response(mf))
    retrms = filter(x->Meta.isexpr(x,:call) && x.args[1] == :|, mf.terms.terms)
    length(retrms) > 0 || error("Formula $f has no random-effects terms")
    LMMBase(f, mf, ModelMatrix(mf), y, similar(y), similar(y),
            {string(t.args[3]) for t in retrms},
            {pool(getindex(mf.df,t.args[3])) for t in retrms},
            {lhs2mat(t,mf.df) for t in retrms})
end

## Convert the left-hand side of a random-effects term to a model matrix.
## Special handling for a simple, scalar r.e. term, e.g. (1|g).
lhs2mat(t::Expr,df::DataFrame) = t.args[2] == 1 ? ones(nrow(df),1) :
        ModelMatrix(ModelFrame(Formula(nothing,t.args[2]),df)).m

## fnames(lmb) -> vector of names of grouping factors
fnames(lmb::LMMBase) = lmb.fnms

## grplevels(lmb) -> Vector{Int} : number of levels in each term's grouping factor
grplevels(lmb::LMMBase) = [length(f.pool) for f in lmb.facs]

## isnested(lmb) -> Bool : Are the grouping factors nested?
function isnested(lmb::LMMBase)
    f = lmb.facs
    length(f) == 1 || length(Set(zip(f...))) == maximum(grplevels(lmb))
end

StatsBase.nobs(lmb::LMMBase) = length(lmb.y)

## isscalar(lmb) -> Bool : Are all the random-effects terms scalar?
isscalar(lmb::LMMBase) = all(pvec(lmb) .== 1)

StatsBase.model_response(lmb::LMMBase) = lmb.y

## pvec(lmb) -> Vector{Int} : number of columns in each X of Xs
pvec(lmb::LMMBase) = [size(x,2) for x in lmb.Xs]

## rss(m) -> residual sum of squares
rss(lmb::LMMBase) = NumericExtensions.sumsqdiff(lmb.mu,lmb.y)

##  size(m) -> n, p, q, t (lengths of y, beta, u and # of re terms)
function Base.size(lmb::LMMBase)
    n,p = size(lmb.X.m)
    n,p,sum(grplevels(lmb) .* pvec(lmb)),length(lmb.fnms)
end

## Return a block in the Zt matrix from one term.
function Ztblk(m::Matrix,v)
    nr,nc = size(m)
    nblk = maximum(v)
    NR = nc*nblk                        # number of rows in result
    cf = length(m) < typemax(Int32) ? int32 : int64 # conversion function
    SparseMatrixCSC(NR,nr,
                    cf(cumsum(vcat(1,fill(nc,(nr,))))), # colptr
                    cf(vec(reshape([1:NR],(nc,int(nblk)))[:,v])), # rowval
                    vec(m'))            # nzval
end
Ztblk(m::Matrix,v::PooledDataVector) = Ztblk(m,v.refs)

Zt(lmb::LMMBase) = vcat(map(Ztblk,lmb.Xs,lmb.facs)...)

ZXt(lmb::LMMBase) = (zt = Zt(lmb); vcat(zt,convert(typeof(zt),lmb.X.m')))
