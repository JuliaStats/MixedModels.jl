type LinearMixedModel{S<:PLSSolver} <: MixedModel
    REML::Bool
    X::ModelMatrix{Float64}
    Xs::Vector
    Xty::Vector{Float64}
    Zty::Vector
    b::Vector
    f::Formula
    facs::Vector
    fit::Bool
    fnms::Vector                        # names of grouping factors
    mf::ModelFrame
    resid::Vector{Float64}
    s::S
    u::Vector
    y::Vector{Float64}
    β::Vector{Float64}
    λ::Vector
    μ::Vector{Float64}
end

## Convert the left-hand side of a random-effects term to a model matrix.
## Special handling for a simple, scalar r.e. term, e.g. (1|g).
## FIXME: Change this behavior in DataFrames/src/statsmodels/formula.jl
lhs2mat(t::Expr,df::DataFrame) = t.args[2] == 1 ? ones(nrow(df),1) :
        ModelMatrix(ModelFrame(Formula(nothing,t.args[2]),df)).m

function lmm(f::Formula, fr::AbstractDataFrame)
    mf = ModelFrame(f,fr)
    y = convert(Vector{Float64},DataFrames.model_response(mf))
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
    local s
    if length(Xs) == 1
        s = PLSOne(facs[1],Xs[1],X.m')
    else
        Zt = vcat(map(ztblk,Xs,facs)...)
        s = all(p .== 1) ? PLSDiag(Zt,X.m,facs) : PLSGeneral(Zt,X.m,facs)
    end
    LinearMixedModel(false, X, Xs, Xty, Zty, {similar(z) for z in Zty},
        f, facs, false, {string(t.args[3]) for t in retrms}, mf,
        similar(y), s, {similar(z) for z in Zty}, y, similar(Xty),
        {Triangular(eye(pp),:L,false) for pp in p}, similar(y))
end

## Return the Cholesky factor RX or L
Base.cholfact(m::LinearMixedModel,RX::Bool=true) = cholfact(m.s,RX)

##  coef(m) -> current value of beta (as a reference)
StatsBase.coef(m::LinearMixedModel) = m.β

## Condition number
Base.cond(m::LinearMixedModel) = [cond(λ)::Float64 for λ in m.λ]

Base.cor(m::LinearMixedModel) = map(chol2cor,m.λ)

## coeftable(m) -> DataFrame : the coefficients table
function StatsBase.coeftable(m::LinearMixedModel)
    fe = fixef(m)
    se = stderr(m)
    CoefTable(hcat(fe,se,fe./se), ["Estimate","Std.Error","z value"], ASCIIString[])
end

## deviance(m) -> Float64
function StatsBase.deviance(m::LinearMixedModel)
    m.fit || error("model m has not been fit")
    m.REML ? NaN : objective(m)
end

## fit(m) -> m Optimize the objective using BOBYQA from the NLopt package
function StatsBase.fit(m::LinearMixedModel, verbose=false)
    if !m.fit
        th = θ(m); k = length(th)
        opt = NLopt.Opt(hasgrad(m) ? :LD_MMA : :LN_BOBYQA, k)
        NLopt.ftol_rel!(opt, 1e-8)    # relative criterion on deviance
        NLopt.ftol_abs!(opt, 1e-8)    # absolute criterion on deviance
        NLopt.xtol_abs!(opt, 1e-8)    # criterion on all parameter value changes
        NLopt.lower_bounds!(opt, lower(m))
        function obj(x::Vector{Float64}, g::Vector{Float64})
            val = objective!(m,x)
            length(g) == 0 || grad!(g,m)
            val
        end
        if verbose
            count = 0
            function vobj(x::Vector{Float64}, g::Vector{Float64})
                count += 1
                val = objective!(m,x)
                print("f_$count: $(round(val,5)), [")
                showcompact(x[1])
                for i in 2:length(x) print(","); showcompact(x[i]) end
                println("]")
                length(g) == 0 || grad!(g,m)
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

## for compatibility with lme4 and nlme
fixef(m::LinearMixedModel) = m.β

## fnames(m) -> vector of names of grouping factors
fnames(m::LinearMixedModel) = m.fnms

## overwrite g with the gradient (assuming that objective! has already been called)
function grad!(g,m::LinearMixedModel)
    hasgrad(m) || error("gradient evaluation not provided for $(typeof(m))")
    gg = grad(m.s,scale(m,true),m.resid,m.u,m.λ)
    length(gg) == length(g) || throw(DimensionMismatch(""))
    copy!(g,gg)
end

## grplevels(m) -> Vector{Int} : number of levels in each term's grouping factor
grplevels(v::Vector) = [length(f.pool) for f in v]
grplevels(m::LinearMixedModel) = grplevels(m.facs)

hasgrad(m::LinearMixedModel) = false
hasgrad(m::LinearMixedModel{PLSOne}) = true

isfit(m::LinearMixedModel) = m.fit

isnested(v::Vector) = length(v) == 1 || length(Set(zip(v...))) == maximum(grplevels(v))
isnested(m::LinearMixedModel) = isnested(m.facs)

## isscalar(m) -> Bool : Are all the random-effects terms scalar?
function isscalar(m::LinearMixedModel)
    for x in m.Xs
        size(x,1) > 1 && return false
    end
    true
end

## FixME: Change the definition so that one choice is for the combined L and RX
Base.logdet(m::LinearMixedModel,RX::Bool=true) = logdet(m.s,RX)

## lower(m) -> Vector{Float64} : vector of lower bounds for the theta parameters
lower(m::LinearMixedModel) = vcat(map(lower,m.λ)...)
function lower(x::Triangular)
    (s = size(x,1)) == 1 && return [0.]
    res = fill(-Inf,s*(s+1)>>1)
    k = 1                               # position in res
    for j in s:-1:1
        res[k] = 0.
        k += j
    end
    res
end

## likelihood ratio tests
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

## nobs(m) -> n : Length of the response vector
StatsBase.nobs(m::LinearMixedModel) = length(m.y)

## npar(m) -> P : total number of parameters to be fit
npar(m::LinearMixedModel) = nθ(m) + length(m.β) + 1

## nθ(m) -> n : length of the theta vector
## FIXME: make this generic to apply to a general λ
nθ(m::LinearMixedModel) = sum([n*(n+1)>>1 for (m,n) in map(size,m.λ)])

## objective(m) -> deviance or REML criterion according to m.REML
function objective(m::LinearMixedModel)
    n,p = size(m)
    REML = m.REML
    fn = float64(n - (REML ? p : 0))
    logdet(m,false) + fn*(1.+log(2π*pwrss(m)/fn)) + (REML ? logdet(m) : 0.)
end

## objective!(m,θ) -> install new θ parameters and evaluate the objective.
function objective!(m::LinearMixedModel,θ::Vector{Float64})
    update!(m.s,θ!(m,θ))
    for (λ,u,Zty) in zip(m.λ,m.u,m.Zty)
        Ac_mul_B!(λ,copy!(u,Zty))
    end
    plssolve!(m.s,m.u,copy!(m.β,m.Xty))
    updateμ!(m)
    objective(m)
end

## pwrss(lmb) : penalized, weighted residual sum of squares
function pwrss(m::LinearMixedModel)
    s = rss(m)
    for u in m.u, ui in u
        s += abs2(ui)
    end
    s
end

##  ranef(m) -> vector of matrices of random effects on the original scale
##  ranef(m,true) -> vector of matrices of random effects on the U scale
ranef(m::LinearMixedModel, uscale=false) = uscale ? m.u : m.b

##  reml!(m,v=true) -> m : Set m.REML to v.  If m.REML is modified, unset m.fit
function reml!(m::LinearMixedModel,v::Bool=true)
    if m.REML != v
        m.REML = v
        m.fit = false
    end
    m
end

## rss(m) -> residual sum of squares
rss(m::LinearMixedModel) = sum(Abs2Fun(),m.resid)

## scale(m,true) -> estimate, s^2, of the squared scale parameter
function Base.scale(m::LinearMixedModel, sqr=false)
    n,p = size(m.X.m)
    ssqr = pwrss(m)/float64(n - (m.REML ? p : 0))
    sqr ? ssqr : sqrt(ssqr)
end

##  size(m) -> n, p, q, t (lengths of y, beta, u and # of re terms)
function Base.size(m::LinearMixedModel)
    n,p = size(m.X.m)
    n,p,mapreduce(length,+,m.u),length(m.fnms)
end

function Base.show(io::IO, m::LinearMixedModel)
    fit(m)
    n,p,q,k = size(m)
    REML = m.REML
    @printf(io, "Linear mixed model fit by %s\n", REML ? "REML" : "maximum likelihood")

    oo = objective(m)
    if REML
        @printf(io, " REML criterion: %f", oo)
    else
        @printf(io, " logLik: %f, deviance: %f", -oo/2., oo)
    end
    println(io); println(io)

    @printf(io, " Variance components:\n                Variance    Std.Dev.\n")
    stdm = std(m)
    fnms = vcat(m.fnms,"Residual")
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

## sqrlenu(m) -> squared length of m.u (the penalty in the PLS problem)
function sqrlenu(m::LinearMixedModel)
    s = 0.
    for u in m.u, ui in u
        s+=abs2(ui)
    end
    s
end

## std(m) -> Vector{Vector{Float64}} estimated standard deviations of variance components
Base.std(m::LinearMixedModel) = scale(m)*push!([rowlengths(λ) for λ in m.λ],[1.])

## stderr(m) -> standard errors of fixed-effects parameters
StatsBase.stderr(m::LinearMixedModel) = sqrt(diag(vcov(m)))

## update m.μ and return the residual sum of squares
function updateμ!(m::LinearMixedModel)
    μ = A_mul_B!(m.μ, m.X.m, m.β) # initialize μ to Xβ
    for (ff,λ,u,b,x) in zip(m.facs,m.λ,m.u,m.b,m.Xs)
        rr = ff.refs
        A_mul_B!(b,λ,u)         # overwrite b by λ*u
        for i in 1:length(μ)    # @inbounds this loop if successful
            μ[i] += dot(b[:,rr[i]],x[:,i])
        end
    end
    s = 0.
    @inbounds for i in 1:length(μ)
        rr = m.resid[i] = m.y[i] - μ[i]
        s += abs2(rr)
    end
    s
end

## vcov(m) -> estimated variance-covariance matrix of the fixed-effects parameters
StatsBase.vcov(m::LinearMixedModel) = scale(m,true) * inv(cholfact(m.s))

zt(m::LinearMixedModel) = vcat(map(ztblk,m.Xs,m.facs)...)
zxt(m::LinearMixedModel) = (Zt = zt(m); vcat(Zt,convert(typeof(Zt),m.X.m')))

## θ(m) -> θ : extract the covariance parameters as a vector
θ(m::LinearMixedModel) = vcat(map(ltri,m.λ)...)

## θ!(m,theta) -> m : install new values of the covariance parameters
function θ!(m::LinearMixedModel,th::Vector)
    length(th) == nθ(m) || throw(DimensionMismatch(""))
    pos = 0
    for λ in m.λ
        s = size(λ,1)
        for j in 1:s, i in j:s
            λ.data[i,j] = th[pos += 1]
        end
    end
    m.λ
end

function bootstrap!(res::AbstractArray{Float64,2}, m::LinearMixedModel, f::Function, uvals::Bool=true)
    mm = deepcopy(m)
    vv = f(mm)
    isa(vv, Vector{Float64}) && length(vv) == size(res,1) ||
        error("f must return a Vector{Float64} of length $(size(res,1))")
    if uvals
        error("code not yet written")
    else
        d = IsoNormal(mm.μ,scale(mm))
        eyes = [eye(size(l,1)) for l in mm.λ]
        for i in 1:size(res,2)
            rand!(d,mm.y)               # simulate a response vector
            mm.fit = false
            for j in 1:length(λ)
                copy!(mm.λ[j].data,eyes[j])
            end
            res[:,i] = f(fit(mm,true))
        end
    end
    res
end
