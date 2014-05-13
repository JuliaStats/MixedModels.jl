## Base implementations of methods for the LinearMixedModel abstract type

## Convert the left-hand side of a random-effects term to a model matrix.
## Special handling for a simple, scalar r.e. term, e.g. (1|g).
lhs2mat(t::Expr,df::DataFrame) = t.args[2] == 1 ? ones(nrow(df),1) :
        ModelMatrix(ModelFrame(Formula(nothing,t.args[2]),df)).m

## Information common to all LinearMixedModel types
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
    retrms = filter(x->Meta.isexpr(x,:call)&& x.args[1] == :|, mf.terms.terms)
    length(retrms) > 0 || error("Formula $f has no random-effects terms")
    LMMBase(f, mf, ModelMatrix(mf), y, similar(y), similar(y),
            {string(t.args[3]) for t in retrms},
            {pool(getindex(mf.df,t.args[3])) for t in retrms},
            {lhs2mat(t,mf.df) for t in retrms})
end

grplevels(lmb::LMMBase) = [length(f.pool) for f in lmb.facs]
grplevels(m::LinearMixedModel) = grplevels(m.lmb)

pvec(lmb::LMMBase) = [size(x,2) for x in lmb.Xs]

##  size(m) -> n, p, q, t (lengths of y, beta, u and # of re terms)
function size(lmb::LMMBase)
    n,p = size(lmb.X.m)
    n,p,sum(grplevels(lmb) .* pvec(lmb)),length(lmb.fnms)
end
size(m::LinearMixedModel) = size(m.lmb)

## isscalar(m) -> Bool : Are all the random-effects terms scalar?
isscalar(lmb::LMMBase) = all(pvec(lmb) .== 1)
isscalar(m::LinearMixedModel) = isscalar(m.lmb)

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

## fit(m) -> m Optimize the objective using BOBYQA from the NLopt package
function StatsBase.fit(m::LinearMixedModel, verbose=false)
    if !isfit(m)
        th = theta(m); k = length(th)
        opt = NLopt.Opt(:LN_BOBYQA, k)
        NLopt.ftol_abs!(opt, 1e-6)    # criterion on deviance changes
        NLopt.xtol_abs!(opt, 1e-6)    # criterion on all parameter value changes
        NLopt.lower_bounds!(opt, lower(m))
        function obj(x::Vector{Float64}, g::Vector{Float64})
            length(g) == 0 || error("gradient evaluations are not provided")
            objective(solve!(theta!(m,x),true))
        end
        if verbose
            count = 0
            function vobj(x::Vector{Float64}, g::Vector{Float64})
                length(g) == 0 || error("gradient evaluations are not provided")
                count += 1
                val = objective(solve!(theta!(m,x),true))
                print("f_$count: $(round(val,5)), [")
                showcompact(x[1])
                for i in 2:length(x) print(","); showcompact(x[i]) end
                println("]")
                val
            end
            NLopt.min_objective!(opt, vobj)
        else
            NLopt.min_objective!(opt, obj)
        end
        fmin, xmin, ret = NLopt.optimize(opt, th)
        if verbose println(ret) end
        m.fit = true
    end
    m
end

##  coef(m) -> current value of beta (can be a reference)
StatsBase.coef(m::LinearMixedModel) = m.beta

## coeftable(m) -> DataFrame : the coefficients table
## FIXME Create a type with its own show method for this type of table
function StatsBase.coeftable(m::LinearMixedModel)
    fe = fixef(m); se = stderr(m)
    CoefTable(hcat(fe,se,fe./se), ["Estimate","Std.Error","z value"], ASCIIString[])
end

## deviance(m) -> Float64
StatsBase.deviance(m::LinearMixedModel) = m.fit && !m.REML ? objective(m) : NaN
        
## fixef(m) -> current value of beta (can be a reference)
fixef(m::LinearMixedModel) = m.beta

## fnames(m) -> names of grouping factors
fnames(m::LinearMixedModel) = m.lmb.fnms

##  isfit(m) -> Bool - Has the model been fit?
isfit(m::LinearMixedModel) = m.fit

function lrt(mods::LinearMixedModel...)
    if (nm = length(mods)) <= 1
        error("at least two models are required for an lrt")
    end
    m1 = mods[1]; n = nobs(m1)
    for i in 2:nm
        if nobs(mods[i]) != n
            error("number of observations must be constant across models")
        end
    end
    mods = mods[sortperm([npar(m)::Int for m in mods])]
    df = [npar(m)::Int for m in mods]
    dev = [deviance(m)::Float64 for m in mods]
    csqr = [NaN, [(dev[i-1]-dev[i])::Float64 for i in 2:nm]]
    pval = [NaN, [ccdf(Chisq(df[i]-df[i-1]),csqr[i])::Float64 for i in 2:nm]]
    DataFrame(Df = df, Deviance = dev, Chisq=csqr,pval=pval)
end


nobs(m::LinearMixedModel) = size(m)[1]

npar(m::LinearMixedModel) = length(theta(m)) + length(coef(m)) + 1

## objective(m) -> deviance or REML criterion according to m.REML
function objective(m::LinearMixedModel)
    n,p,q,k = size(m); fn = float64(n - (m.REML ? p : 0))
    logdet(m,false) + fn*(1.+log(2.pi*pwrss(m)/fn)) + (m.REML ? logdet(m) : 0.)
end

## pwrss(m) -> penalized, weighted residual sum of squares
pwrss(m::LinearMixedModel) = rss(m) + sqrlenu(m)

##  reml!(m,v=true) -> m : Set m.REML to v.  If m.REML is modified, unset m.fit
function reml!(m::LinearMixedModel,v=true)
    v == m.REML && return m
    m.REML = v; m.fit = false
    m
end

## rss(m) -> residual sum of squares
rss(lmb::LMMBase) = sumsqdiff(lmb.mu,lmb.y)
rss(m::LinearMixedModel) = rss(m.lmb)

## scale(m) -> estimate, s, of the scale parameter
## scale(m,true) -> estimate, s^2, of the squared scale parameter
function Base.scale(m::LinearMixedModel, sqr=false)
    n,p = size(m); ssqr = pwrss(m)/float64(n - (m.REML ? p : 0))
    sqr ? ssqr : sqrt(ssqr)
end

function Base.show(io::IO, m::LinearMixedModel)
    fit(m); n, p, q, k = size(m); REML = m.REML
    @printf(io, "Linear mixed model fit by %s\n", REML ? "REML" : "maximum likelihood")

    oo = objective(m)
    if REML
        @printf(io, " REML criterion: %f", objective(m))
    else
        @printf(io, " logLik: %f, deviance: %f", -oo/2., oo)
    end
    println(io); println(io)

    @printf(io, " Variance components:\n                Variance    Std.Dev.\n")
    stdm = std(m); fnms = vcat(m.lmb.fnms,"Residual")
    for i in 1:length(fnms)
        si = stdm[i]
        print(io, " ", rpad(fnms[i],12))
        @printf(io, " %10f  %10f\n", abs2(si[1]), si[1])
        for j in 2:length(si)
            @printf(io, "             %10f  %10f\n", abs2(si[j]), si[j])
        end
    end
    gl = grplevels(m)
    @printf(io," Number of obs: %d; levels of grouping factors: %d", n, gl[1])
    for l in gl[2:end] @printf(io, ", %d", l) end
    println(io)
    @printf(io,"\n  Fixed-effects parameters:\n")
    show(io,coeftable(m))
end

## stderr(m) -> standard errors of fixed-effects parameters
StatsBase.stderr(m::LinearMixedModel) = sqrt(diag(vcov(m)))

## vcov(m) -> estimated variance-covariance matrix of the fixed-effects parameters
StatsBase.vcov(m::LinearMixedModel) = scale(m,true) * inv(cholfact(m))
