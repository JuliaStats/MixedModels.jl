## Information common to all LinearMixedModel types
## Many methods for LinearMixedModel pass through to this type
type LMMBase
    f::Formula
    mf::ModelFrame
    X::ModelMatrix{Float64}
    y::Vector{Float64}
    μ::Vector{Float64}
    fnms::Vector                        # names of grouping factors
    facs::Vector
    Xs::Vector
    Xty::Vector{Float64}
    Zty::Vector
    β::Vector{Float64}
    λ::Vector
    u::Vector
    REML::Bool
    fit::Bool
end

## Convert the left-hand side of a random-effects term to a model matrix.
## Special handling for a simple, scalar r.e. term, e.g. (1|g).
lhs2mat(t::Expr,df::DataFrame) = t.args[2] == 1 ? ones(nrow(df),1) :
        ModelMatrix(ModelFrame(Formula(nothing,t.args[2]),df)).m

function LMMBase(f::Formula, fr::AbstractDataFrame)
    mf = ModelFrame(f,fr)
    y = convert(Vector{Float64},model_response(mf))
    retrms = filter(x->Meta.isexpr(x,:call) && x.args[1] == :|, mf.terms.terms)
    length(retrms) > 0 || error("Formula $f has no random-effects terms")
    X = ModelMatrix(mf)
    Xty = X.m'y
    facs = {pool(getindex(mf.df,t.args[3])) for t in retrms}
    Xs = {lhs2mat(t,mf.df)' for t in retrms} # transposed model matrices
    p = Int[size(x,1) for x in Xs]
    l = Int[length(f.pool) for f in facs]
    Zty = {zeros(pp,ll) for (pp,ll) in zip(p,l)}
    for (x,ff,zty,pp) in zip(Xs,facs,Zty,p)
        for (j,jj) in enumerate(ff.refs)
            for i in 1:pp
                zty[i,jj] += y[j] * x[i,j]
            end
        end
    end
    LMMBase(f, mf, X, y, similar(y),
            {string(t.args[3]) for t in retrms},
            facs, Xs, Xty, Zty, similar(Xty),
            {Triangular(eye(pp),:L,false) for pp in p},
            {similar(z) for z in Zty}, false, false)
end

##  coef(lmb) -> current value of beta (as a reference)
StatsBase.coef(lmb::LMMBase) = lmb.β

##  fixef(lmb) -> current value of beta (as a reference)
fixef(lmb::LMMBase) = lmb.β

## fnames(lmb) -> vector of names of grouping factors
fnames(lmb::LMMBase) = lmb.fnms

## grplevels(lmb) -> Vector{Int} : number of levels in each term's grouping factor
grplevels(lmb::LMMBase) = [length(f.pool) for f in lmb.facs]

## isnested(lmb) -> Bool : Are the grouping factors nested?
function isnested(lmb::LMMBase)
    f = lmb.facs
    length(f) == 1 || length(Set(zip(f...))) == maximum(grplevels(lmb))
end

##  isfit(m) -> Bool - Has the model been fit?
isfit(m::LMMBase) = m.fit

## isscalar(lmb) -> Bool : Are all the random-effects terms scalar?
function isscalar(lmb::LMMBase)
    for x in lmb.Xs
        size(x,1) > 1 && return false
    end
    true
end

## update lmb.μ
function updateμ!(lmb::LMMBase)
    μ = A_mul_B!(lmb.μ, lmb.X.m, lmb.β) # initialize μ to Xβ
    for (ff,λ,u,x) in zip(lmb.facs,lmb.λ,lmb.u,lmb.Xs)
        rr = ff.refs
        bb = λ * u
        if size(bb,1) == 1
            bb = vec(bb)
            xx = vec(x)
            for i in 1:length(μ)
                μ[i] += xx[i] * bb[rr[i]]
            end
        else
            for i in 1:length(μ)
                μ[i] += dot(sub(x,:,i),sub(bb,:,int(rr[i])))
            end
        end
    end
    ssqdiff(μ,lmb.y)
end

# pattern of lower bounds for lower triangle of size n
lower(n::Integer) = vcat({vcat(0.,fill(-Inf,k)) for k in n-1:-1:0}...)

# lower(lmb) -> Vector{Float64} : vector of lower bounds for the theta parameters
lower(lmb::LMMBase) = vcat({lower(size(x,1)) for x in lmb.Xs}...)

## model_response(lmb) -> y : returns a reference to the response vector
#StatsBase.model_response(lmb::LMMBase) = lmb.y  # model_response is defined in DataFrames

## nobs(lmb) -> n : Length of the response vector
StatsBase.nobs(lmb::LMMBase) = length(lmb.y)

## npar(lmb) -> n : total number of parameters to be fit
npar(lmb::LMMBase) = nθ(lmb) + length(lmb.β) + 1

## nθ(lmb) -> n : length of the theta vector
nθ(lmb::LMMBase) = sum([n*(n+1)>>1 for (m,n) in map(size,lmb.λ)])

## pwrss(lmb) : penalized, weighted residual sum of squares
function pwrss(lmb::LMMBase)
    s = rss(lmb)
    for u in lmb.u, ui in u
        s += abs2(ui)
    end
    s
end

function ssqdiff{T<:Number}(a::Vector{T},b::Vector{T})
    (n = length(a)) == length(b) || error("Dimension mismatch")
    s = zero(T)
    @simd for i in 1:n
        s += abs2(a[i]-b[i])
    end
    s
end

##  ranef(m) -> vector of matrices of random effects on the original scale
##  ranef(m,true) -> vector of matrices of random effects on the U scale
function ranef(lmb::LMMBase, uscale=false)
    uscale && return lmb.u
    [λ * u for (λ,u) in zip(lmb.λ,lmb.u)]
end

##  reml!(lmb,v=true) -> lmb : Set lmb.REML to v.  If lmb.REML is modified, unset m.fit
function reml!(lmb::LMMBase,v::Bool=true)
    if lmb.REML != v
        lmb.REML = v
        lmb.fit = false
    end
    lmb
end

## rss(m) -> residual sum of squares
rss(lmb::LMMBase) = ssqdiff(lmb.μ,lmb.y)

## scale(m) -> estimate, s, of the scale parameter
## scale(m,true) -> estimate, s^2, of the squared scale parameter
function Base.scale(lmb::LMMBase, sqr=false)
    n,p = size(lmb)
    ssqr = pwrss(lmb)/float64(n - (lmb.REML ? p : 0))
    sqr ? ssqr : sqrt(ssqr)
end

##  size(m) -> n, p, q, t (lengths of y, beta, u and # of re terms)
function Base.size(lmb::LMMBase)
    n,p = size(lmb.X.m)
    n,p,mapreduce(length,+,lmb.u),length(lmb.fnms)
end

## sqrlenu(lmb) -> squared length of lmb.u (the penalty in the PLS problem)
function sqrlenu(lmb::LMMBase)
    s = 0.
    for u in lmb.u, ui in u
        s+=abs2(ui)
    end
    s
end

## rowlengths(m) -> v : return a vector of the row lengths
rowlengths(m::Matrix{Float64}) = [norm(sub(m,i,:))::Float64 for i in 1:size(m,1)]
rowlengths(t::Triangular{Float64}) = rowlengths(full(t))

## std(m) -> Vector{Vector{Float64}} estimated standard deviations of variance components
Base.std(lmb::LMMBase) = scale(lmb)*push!([rowlengths(λ) for λ in lmb.λ],[1.])

## Return a block in the Zt matrix from one term.
function Ztblk(m::Matrix,v)
    nr,nc = size(m)
    nblk = maximum(v)
    NR = nr*nblk                        # number of rows in result
    cf = length(m) < typemax(Int32) ? int32 : int64 # conversion function
    SparseMatrixCSC(NR,nc,
                    cf(cumsum(vcat(1,fill(nr,(nc,))))), # colptr
                    cf(vec(reshape([1:NR],(nr,int(nblk)))[:,v])), # rowval
                    vec(m))            # nzval
end
Ztblk(m::Matrix,v::PooledDataVector) = Ztblk(m,v.refs)

Zt(lmb::LMMBase) = vcat(map(Ztblk,lmb.Xs,lmb.facs)...)

ZXt(lmb::LMMBase) = (zt = Zt(lmb); vcat(zt,convert(typeof(zt),lmb.X.m')))

## ltri(m) -> v : extract the lower triangle as a vector
function ltri(m::Matrix)
    n = size(m,1)
    n == 1 && return copy(vec(m))
    res = Array(eltype(m),n*(n+1)>>1)
    pos = 0
    for j in 1:n, i in j:n
        res[pos += 1] = m[i,j]
    end
    res
end
function ltri(t::Triangular)
    t.uplo == 'L' || error("Triangular matrix must be lower triangular")
    ltri(t.UL)
end

## θ(lmb) -> θ : extract the covariance parameters as a vector
θ(lmb::LMMBase) = vcat(map(ltri,lmb.λ)...)

## θ!(lmb,theta) -> lmb : install new values of the covariance parameters
function θ!(lmb::LMMBase,th::Vector)
    length(th) == nθ(lmb) || error("Dimension mismatch")
    pos = 0
    for λ in lmb.λ
        s = size(λ,1)
        for j in 1:s, i in j:s
            λ.UL[i,j] = th[pos += 1]
        end
    end
    lmb.λ
end

## Make a version of this and of Ztblk that overwrites storage

## λtZt(lmb) -> λtZt : extract λ'Z' as a sparse matrix
λtZt(lmb::LMMBase) = vcat(map(Ztblk, [λ'*Xs for (λ,Xs) in zip(lmb.λ,lmb.Xs)], lmb.facs)...)

## install λ'Z'y in u and Xty in β
function λtZty!(lmb::LMMBase)
    for (λ,Zty,u) in zip(lmb.λ,lmb.Zty,lmb.u)
        A_mul_B!(λ,copy!(u,Zty))
    end
    copy!(lmb.β,lmb.Xty)
    lmb
end
