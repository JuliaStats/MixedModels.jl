"Summary of optimization using the NLopt package"
type OptSummary
    initial::Vector{Float64}
    final::Vector{Float64}
    fmin::Float64
    feval::Int
    geval::Int
    optimizer::Symbol
end

OptSummary() = OptSummary(Float64[],Float64[],NaN,-1,-1,:default)

"Representation of a linear mixed-effects model"
type LinearMixedModel{S<:PLSSolver} <: MixedModel
    REML::Bool
    X::ModelMatrix{Float64}
    Xs::Vector
    Xty::Vector{Float64}
    Ztblks::Vector
    Zty::Vector
    b::Vector
    f::Formula
    facs::Vector
    fit::Bool      # can probably remove and use fields of opt instead
    fnms::Vector   # names of grouping factors
    gradblk::Vector{Matrix{Float64}}
    mf::ModelFrame
    resid::Vector{Float64}
    s::S
    u::Vector
    uβ::Vector{Float64} # concatenation of spherical random effects and fixed-effects
    y::Vector{Float64}
    λ::Vector
    μ::Vector{Float64}
    opt::OptSummary
end

"Create a linear mixed-effects model from a formula and data frame"
function lmm(f::Formula, fr::AbstractDataFrame)
    mf = ModelFrame(f,fr)
    X = ModelMatrix(mf)
    y = convert(Vector{Float64},DataFrames.model_response(mf))
    Xty = X.m'y
                                        # process the random-effects terms
    retrms = filter(x->Meta.isexpr(x,:call) && x.args[1] == :|, mf.terms.terms)
    length(retrms) > 0 || error("Formula $f has no random-effects terms")
    Xs = [lhs2mat(t,mf.df)' for t in retrms] # transposed model matrices
    p = Int[size(x,1) for x in Xs]
    λ = Any[pp == 1 ? PDScalF(1.,1) : PDLCholF(cholfact(eye(pp),:L)) for pp in p]
    grps = Any[t.args[3] for t in retrms]
    ugrps = unique(grps)
    if length(ugrps) < length(grps)     # amalgamate terms with the same grouping factor
        for g in ugrps
            ii = find(grps .== g)
            length(ii) == 1 && continue
            iii = copy(ii)
            i1 = shift!(ii)
            p[i1] = sum(p[iii])
            deleteat!(p,ii)
            Xs[i1] = vcat(Xs[iii]...)
            deleteat!(Xs,ii)
            λ[i1] = amalgamate(λ[iii])
            deleteat!(λ,ii)
            deleteat!(grps,ii)
        end
    end
    facs = [getindex(mf.df,g) for g in ugrps]
    map!(x->isa(x,PooledDataArray) ? x : pool(x), facs)
    l = Int[length(f.pool) for f in facs]

    q = sum(p .* l)
    uβ = zeros(q + size(X,2))
    Zty = [zeros(pp,ll) for (pp,ll) in zip(p,l)]
    u = Any[]
    offset = 0
    for (x,ff,zty) in zip(Xs,facs,Zty)
        push!(u,ContiguousView(uβ, offset, size(zty)))
        offset += length(zty)
        for (j,jj) in enumerate(ff.refs)
            for i in 1:size(zty,1)
                zty[i,jj] += y[j] * x[i,j]
            end
        end
    end
    local s
    if length(facs) == 1
        s = PLSOne(facs[1],Xs[1],X.m')
    elseif isnested(facs)
        s = PLSNested(facs,Xs,X.m')
    elseif length(facs) == 2 && 2countnz(crosstab(facs[1],facs[2])) > l[1]*l[2]
        ## use PLSTwo for two grouping factors and the crosstab density > 0.5
        s = PLSTwo(facs,Xs,X.m')
    else
        Zt = vcat(map(ztblk,Xs,facs)...)
        s = all(p .== 1) ? PLSDiag(Zt,X.m,facs) : PLSGeneral(Zt,X.m,facs)
    end
    LinearMixedModel(false, X, Xs, Xty, map(ztblk,Xs,facs), Zty,
                     map(zeros, u), f, facs, false, map(string,ugrps),
                     Matrix{Float64}[zeros(k,k) for k in p], mf, similar(y),
                     s, u, uβ, y, λ, similar(y),OptSummary())
end

"Return the Cholesky factor RX or L from a LinearMixedModel"
Base.cholfact(m::LinearMixedModel,RX::Bool=true) = cholfact(m.s,RX)

"coef(m) -> current value of beta (as a reference)"
StatsBase.coef(m::LinearMixedModel) = fixef(m)

termnames(term::Symbol, col) = [string(term)]
function termnames(term::Symbol, col::PooledDataArray)
    levs = levels(col)
    [string(term, levs[i]) for i in 2:length(levs)]
end

## Temporary copy until change in DataFrames is merged and in a new release.
## coefnames(m) -> return a vector of coefficient names
function DataFrames.coefnames(m::LinearMixedModel)
    fr = m.mf
    if fr.terms.intercept
        vnames = UTF8String["(Intercept)"]
    else
        vnames = UTF8String[]
    end
    # Need to only include active levels
    for term in fr.terms.terms
        if isa(term, Expr)
            if term.head == :call && term.args[1] == :|
                continue                # skip random-effects terms
            elseif term.head == :call && term.args[1] == :&
                a = term.args[2]
                b = term.args[3]
                for lev1 in termnames(a, fr.df[a]), lev2 in termnames(b, fr.df[b])
                    push!(vnames, string(lev1, "&", lev2))
                end
            else
                error("unrecognized term $term")
            end
        else
            append!(vnames, termnames(term, fr.df[term]))
        end
    end
    return vnames
end

Base.cond(m::LinearMixedModel) = [cond(λ)::Float64 for λ in m.λ]

Base.cor(m::LinearMixedModel) = map(chol2cor,m.λ)

function StatsBase.coeftable(m::LinearMixedModel)
    fe = fixef(m)
    se = stderr(m)
    CoefTable(hcat(fe,se,fe./se), ["Estimate","Std.Error","z value"], coefnames(m))
end

function StatsBase.deviance(m::LinearMixedModel)
    m.fit || error("model m has not been fit")
    m.REML ? NaN : objective(m)
end

"`fit(m)` -> m Optimize the objective using an NLopt optimizer"
function StatsBase.fit(m::LinearMixedModel, verbose::Bool=false, optimizer::Symbol=:default)
    if !m.fit
        th = θ(m); k = length(th)
        if optimizer == :default
            optimizer = hasgrad(m) ? :LD_MMA : :LN_BOBYQA
        end
        opt = NLopt.Opt(optimizer, k)
        NLopt.ftol_rel!(opt, 1e-12)   # relative criterion on deviance
        NLopt.ftol_abs!(opt, 1e-8)    # absolute criterion on deviance
        NLopt.xtol_abs!(opt, 1e-10)   # criterion on parameter value changes
        NLopt.lower_bounds!(opt, lower(m))
        feval = 0
        geval = 0
        function obj(x::Vector{Float64}, g::Vector{Float64})
            feval += 1
            val = objective!(m,x)
            if length(g) == length(x)
                geval += 1
                grad!(g,m)
            end
            val
        end
        if verbose
            function vobj(x::Vector{Float64}, g::Vector{Float64})
                feval += 1
                val = objective!(m,x)
                print("f_$feval: $(round(val,5)), [")
                showcompact(x[1])
                for i in 2:length(x) print(","); showcompact(x[i]) end
                println("]")
                if length(g) == length(x)
                    geval += 1
                    grad!(g,m)
                end
                val
            end
            NLopt.min_objective!(opt, vobj)
        else
            NLopt.min_objective!(opt, obj)
        end
        fmin, xmin, ret = NLopt.optimize(opt, th)
        # very small parameter values often should be set to zero
        xmin1 = copy(xmin)
        modified = false
        for i in eachindex(xmin1)
            if 0. < abs(xmin1[i]) < 1.e-5
                modified = true
                xmin1[i] = 0.
            end
        end
        if modified && (ff = objective!(m,xmin1)) < fmin
            fmin = ff
            copy!(xmin,xmin1)
        end
        m.opt = OptSummary(th,xmin,fmin,feval,geval,optimizer)
        if verbose println(ret) end
        m.fit = true
    end
    m
end

"Return the vector of fixed-effects coefficients"
function fixef(m::LinearMixedModel)
    ppq = length(m.uβ)
    p = length(m.Xty)
    m.uβ[(ppq - p + 1):ppq]
end

"`fnames(m)` -> vector of names of grouping factors"
fnames(m::LinearMixedModel) = m.fnms

"Return Λ as a sparse triangular matrix"
function Λ(m::LinearMixedModel)
    vv = SparseMatrixCSC{Float64,SuiteSparse_long}[]
    for i in 1:length(m.λ)
        nl = length(m.facs[i].pool)
        ll = m.λ[i]
        p = size(ll,1)
        if p == 1
            isa(ll,PDScalF) || error("1×1 λ section should be a PDScalF type")
            push!(vv,ll.s .* speye(nl))
        else
            sfll = sparse(full(ll))
            push!(vv,blkdiag([sfll for _ in 1:nl]...))
        end
    end
    ltri(blkdiag(vv...))
end

"overwrite g with the gradient (assuming that objective! has already been called)"
function grad!(g,m::LinearMixedModel)
    hasgrad(m) || error("gradient evaluation not provided for $(typeof(m))")
    ## overwrite b with -Zt*resid/scale(m,true)
    mult = -1./scale(m,true)
    for i in 1:length(m.b)
        A_mul_B!(mult,m.Ztblks[i],m.resid,0.,vec(m.b[i]))
        BLAS.gemm!('N','T',1.,m.u[i],m.b[i],0.,m.gradblk[i])
    end
    grad!(m.gradblk,m.s,m.λ)
    offset = 0
    for i in 1:length(m.gradblk)
        ll = m.λ[i]
        nth = nθ(ll)
        grdcmp!(view(g,offset+(1:nth)),ll,m.gradblk[i])
        offset += nth
    end
    g
end

## grplevels(m) -> Vector{Int} : number of levels in each term's grouping factor
grplevels(v::Vector) = [length(f.pool) for f in v]
grplevels(m::LinearMixedModel) = grplevels(m.facs)

hasgrad(::LinearMixedModel) = false
## No, these can't be moved to plsone.jl and plstwo.jl because of the order of file inclusion.
hasgrad(::LinearMixedModel{PLSOne}) = true
hasgrad(::LinearMixedModel{PLSTwo}) = true
#hasgrad(::LinearMixedModel{PLSNested}) = true # bugs to fix here

isfit(m::LinearMixedModel) = m.fit

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
    csqr = unshift!([(dev[i-1]-dev[i])::Float64 for i in 2:nm],NaN)
    pval = unshift!([ccdf(Chisq(df[i]-df[i-1]),csqr[i])::Float64 for i in 2:nm],NaN)
    DataFrame(Df = df, Deviance = dev, Chisq=csqr,pval=pval)
end

## nobs(m) -> n : Length of the response vector
StatsBase.nobs(m::LinearMixedModel) = length(m.y)

## npar(m) -> P : total number of parameters to be fit
npar(m::LinearMixedModel) = nθ(m) + length(m.Xty) + 1

## nθ(m) -> n : length of the theta vector
function nθ(m::LinearMixedModel)
    s = 0
    for ll in m.λ
        s += nθ(ll)
    end
    s
end

## objective(m) -> deviance or REML criterion according to m.REML
function objective(m::LinearMixedModel)
    n,p = size(m)
    REML = m.REML
    fn = Float64(n - (REML ? p : 0))
    logdet(m,false) + fn*(1.+log(2π*pwrss(m)/fn)) + (REML ? logdet(m) : 0.)
end

## objective!(m,θ) -> install new θ parameters and evaluate the objective.
function objective!(m::LinearMixedModel,θ::Vector{Float64})
    update!(m.s,θ!(m,θ))
    for (λ,u,Zty) in zip(m.λ,m.u,m.Zty)
        Ac_mul_B!(λ,copy!(u,Zty))
    end
    p = length(m.Xty)
    copy!(ContiguousView(m.uβ,length(m.uβ)-p,(p,)), m.Xty)
    A_ldiv_B!(m.s,m.uβ)
    updateμ!(m)
    objective(m)
end
objective!(m::LinearMixedModel) = objective!(m,θ(m))

## pwrss(lmb) : penalized, weighted residual sum of squares
pwrss(m::LinearMixedModel) = rss(m) + sqrlenu(m)

##  ranef(m) -> vector of matrices of random effects on the original scale
##  ranef(m,true) -> vector of matrices of random effects on the U scale
function ranef(m::LinearMixedModel, uscale=false)
    uscale && return m.u
    for (λ,b,u) in zip(m.λ,m.b,m.u)
        A_mul_B!(λ,copy!(b,u))         # overwrite b by λ*u
    end
    m.b
end

##  reml!(m,v=true) -> m : Set m.REML to v.  If m.REML is modified, unset m.fit
function reml!(m::LinearMixedModel,v::Bool=true)
    if m.REML != v
        m.REML = v
        m.fit = false
    end
    m
end

## rss(m) -> residual sum of squares
rss(m::LinearMixedModel) = sumabs2(m.resid)

## scale(m,true) -> estimate, s^2, of the squared scale parameter
function Base.scale(m::LinearMixedModel, sqr=false)
    n,p = size(m.X.m)
    ssqr = pwrss(m)/Float64(n - (m.REML ? p : 0))
    sqr ? ssqr : sqrt(ssqr)
end

function Base.show(io::IO, m::LinearMixedModel)
    if !m.fit
        warn("Model has not been fit")
        return nothing
    end
    n,p,q,k = size(m)
    REML = m.REML
    @printf(io, "Linear mixed model fit by %s\n", REML ? "REML" : "maximum likelihood")
    println(io, m.f)
    println(io)

    oo = objective(m)
    if REML
        @printf(io, " REML criterion: %f", oo)
    else
        @printf(io, " logLik: %f, deviance: %f", -oo/2., oo)
    end
    println(io); println(io)

    show(io,VarCorr(m))

    gl = grplevels(m)
    @printf(io," Number of obs: %d; levels of grouping factors: %d", n, gl[1])
    for l in gl[2:end] @printf(io, ", %d", l) end
    println(io)
    @printf(io,"\n  Fixed-effects parameters:\n")
    show(io,coeftable(m))
end

##  size(m) -> n, p, q, t (lengths of y, beta, u and # of re terms)
function Base.size(m::LinearMixedModel)
    n,p = size(m.X.m)
    n,p,length(m.uβ)-p,length(m.fnms)
end

## sqrlenu(m) -> squared length of m.u (the penalty in the PLS problem)
sqrlenu(m::LinearMixedModel) = sumabs2(view(m.uβ,1:mapreduce(length,+,m.u)))

## std(m) -> Vector{Vector{Float64}} estimated standard deviations of variance components
Base.std(m::LinearMixedModel) = scale(m)*push!([rowlengths(λ) for λ in m.λ],[1.])

## stderr(m) -> standard errors of fixed-effects parameters
StatsBase.stderr(m::LinearMixedModel) = sqrt(diag(vcov(m)))

## update m.μ and return the residual sum of squares
function updateμ!(m::LinearMixedModel)
    p = length(m.Xty)
    μ = A_mul_B!(m.μ, m.X.m, ContiguousView(m.uβ,length(m.uβ)-p,(p,))) # initialize μ to Xβ
    for (Zt,λ,b,u) in zip(m.Ztblks,m.λ,m.b,m.u)
        A_mul_B!(λ,copy!(b,u))         # overwrite b by λ*u
        Ac_mul_B!(1.0,Zt,vec(b),1.0,μ)
    end
    s = 0.
    @simd for i in 1:length(μ)
        rr = m.resid[i] = m.y[i] - μ[i]
        s += abs2(rr)
    end
    s
end

type VarCorr                            # a type to isolate the print logic
    λ::Vector
    fnms::Vector
    s::Float64
    function VarCorr(λ::Vector,fnms::Vector,s::Number)
        (k = length(fnms)) == length(λ) || throw(DimensionMismatch(""))
        s >= 0. || error("s must be non-negative")
        for i in 1:k
            isa(λ[i],AbstractPDMatFactor) || error("isa(λ[$i],AbstractPDMatFactor is not true")
            isa(fnms[i],String) || error("fnms must be a vector of strings")
        end
        new(λ,fnms,s)
    end
end
VarCorr(m::LinearMixedModel) = VarCorr(m.λ,m.fnms,scale(m))

function Base.show(io::IO,vc::VarCorr)
    fnms = vcat(vc.fnms,"Residual")
    nmwd = maximum(map(strwidth, fnms))
    write(io, "Variance components:\n")
    stdm = vc.s*push!([rowlengths(λ) for λ in vc.λ],[1.])
    tt = vcat(stdm...)
    vi = showoff(abs2(tt), :plain)
    si = showoff(tt, :plain)
    varwd = 1 + max(length("Variance"), maximum(map(strwidth, vi)))
    stdwd = 1 + max(length("Std.Dev."), maximum(map(strwidth, si)))
    write(io, " "^(2+nmwd))
    write(io, Base.cpad("Variance", varwd))
    write(io, Base.cpad("Std.Dev.", stdwd))
    any(s -> length(s) > 1,stdm) && write(io,"  Corr.")
    println(io)
    cor = [chol2cor(λ) for λ in vc.λ]
    ind = 1
    for i in 1:length(fnms)
        stdmi = stdm[i]
        write(io, ' ')
        write(io, rpad(fnms[i], nmwd))
        write(io, lpad(vi[ind], varwd))
        write(io, lpad(si[ind], stdwd))
        ind += 1
        println(io)
        for j in 2:length(stdmi)
            write(io, lpad(vi[ind], varwd + nmwd + 1))
            write(io, lpad(si[ind], stdwd))
            ind += 1
            for k in 1:(j-1)
                @printf(io, "%6.2f", cor[i][j,1])
            end
            println(io)
        end
    end
end
## vcov(m) -> estimated variance-covariance matrix of the fixed-effects parameters
StatsBase.vcov(m::LinearMixedModel) = scale(m,true) * inv(cholfact(m.s))

zt(m::LinearMixedModel) = vcat(map(ztblk,m.Xs,m.facs)...)
zxt(m::LinearMixedModel) = (Zt = zt(m); vcat(Zt,convert(typeof(Zt),m.X.m')))

## θ(m) -> θ : extract the covariance parameters as a vector
θ(m::LinearMixedModel) = vcat(map(θ,m.λ)...)

## θ!(m,theta) -> m : install new values of the covariance parameters
function θ!(m::LinearMixedModel,th::Vector)
    nth = [nθ(l) for l in m.λ]
    length(th) == sum(nth) || throw(DimensionMismatch(""))
    pos = 0
    for i in 1:length(nth)
        θ!(m.λ[i], view(th,pos + (1:nth[i])))
        pos += nth[i]
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
            for j in 1:length(mm.λ)
                copy!(mm.λ[j].data,eyes[j])
            end
            res[:,i] = f(fit(mm,true))
        end
    end
    res
end
