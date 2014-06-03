type LinearMixedModel <: MixedModel
    lmb::LMMBase
    s::PLSSolver
end

## Delegate methods to the lmb member
Base.size(m::LinearMixedModel) = size(m.lmb)
Base.scale(m::LinearMixedModel) = scale(m.lmb)
Base.scale(m::LinearMixedModel,sqr::Bool) = scale(m.lmb,sqr)

StatsBase.coef(m::LinearMixedModel) = coef(m.lmb)
StatsBase.model_response(m::LinearMixedModel) = model_response(m.lmb)
StatsBase.nobs(m::LinearMixedModel) = nobs(m.lmb)

## methods for generics local to this package
ranef(m::LinearMixedModel,uscale::Bool) = ranef(m.lmb,uscale)
for f in (:fixef, :fnames, :grplevels, :isfit, :isnested, :isscalar,
          :lower, :nθ, :pwrss, :ranef, :rss, :Zt, :ZXt, :θ)
    @eval begin
        $f(m::LinearMixedModel) = $f(m.lmb)
    end
end

## coeftable(m) -> DataFrame : the coefficients table
function StatsBase.coeftable(m::LinearMixedModel)
    fe = fixef(m)
    se = stderr(m)
    CoefTable(hcat(fe,se,fe./se), ["Estimate","Std.Error","z value"], ASCIIString[])
end

## deviance(m) -> Float64
function StatsBase.deviance(m::LinearMixedModel)
    lmb = m.lmb
    lmb.fit || error("model m has not been fit")
    !lmb.REML ? objective(m) : NaN
end

## fit(m) -> m Optimize the objective using BOBYQA from the NLopt package
function StatsBase.fit(m::LinearMixedModel, verbose=false)
    if !isfit(m)
        th = θ(m); k = length(th)
        opt = NLopt.Opt(:LN_BOBYQA, k)
        NLopt.ftol_abs!(opt, 1e-6)    # criterion on deviance changes
        NLopt.xtol_abs!(opt, 1e-6)    # criterion on all parameter value changes
        NLopt.lower_bounds!(opt, lower(m))
        function obj(x::Vector{Float64}, g::Vector{Float64})
            length(g) == 0 || error("gradient evaluations are not provided")
            objective!(m,x)
        end
        if verbose
            count = 0
            function vobj(x::Vector{Float64}, g::Vector{Float64})
                length(g) == 0 || error("gradient evaluations are not provided")
                count += 1
                val = objective!(m,x)
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
        m.lmb.fit = true
    end
    m
end

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

## objective(m) -> deviance or REML criterion according to m.REML
function objective(m::LinearMixedModel)
    n,p,q,k = size(m)
    lmb = m.lmb
    s = m.s
    fn = float64(n - (lmb.REML ? p : 0))
    logdet(s,false) + fn*(1.+log(2π*pwrss(lmb)/fn)) + (lmb.REML ? logdet(s) : 0.)
end

## objective!(m,θ) -> install new θ parameters and evaluate the objective.
function objective!(m::LinearMixedModel,θ::Vector{Float64})
    updateμ!(A_ldiv_B!(update!(m.s,θ!(m.lmb,θ)),m.lmb))
    objective(m)
end

rss(m::LinearMixedModel) = rss(m.lmb)

## scale(m) -> estimate, s, of the scale parameter
## scale(m,true) -> estimate, s^2, of the squared scale parameter
function Base.scale(m::LinearMixedModel, sqr=false)
    n,p = size(m); ssqr = pwrss(m)/float64(n - (m.REML ? p : 0))
    sqr ? ssqr : sqrt(ssqr)
end

function Base.show(io::IO, m::LinearMixedModel)
    fit(m)
    n,p,q,k = size(m)
    lmb = m.lmb
    REML = lmb.REML
    @printf(io, "Linear mixed model fit by %s\n", REML ? "REML" : "maximum likelihood")

    oo = objective(m)
    if REML
        @printf(io, " REML criterion: %f", oo)
    else
        @printf(io, " logLik: %f, deviance: %f", -oo/2., oo)
    end
    println(io); println(io)

    @printf(io, " Variance components:\n                Variance    Std.Dev.\n")
    stdm = std(m.lmb)
    fnms = vcat(m.lmb.fnms,"Residual")
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
StatsBase.vcov(m::LinearMixedModel) = scale(m,true) * inv(cholfact(m.s))
