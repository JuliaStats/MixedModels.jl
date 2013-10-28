## Base implementations of methods for the LinearMixedModel abstract type

## fit(m) -> m Optimization the objective using BOBYQA from the NLopt package
function fit(m::LinearMixedModel, verbose=false)
    if !isfit(m)
        th = theta(m); k = length(th)
        opt = Opt(:LN_BOBYQA, k)
        ftol_abs!(opt, 1e-6)    # criterion on deviance changes
        xtol_abs!(opt, 1e-6)    # criterion on all parameter value changes
        lower_bounds!(opt, lower(m))
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
            min_objective!(opt, vobj)
        else
            min_objective!(opt, obj)
        end
        fmin, xmin, ret = optimize(opt, th)
        if verbose println(ret) end
        m.fit = true
    end
    m
end

##  coef(m) -> current value of beta (can be a reference)
coef(m::LinearMixedModel) = m.beta

## coeftable(m) -> DataFrame : the coefficients table
## FIXME Create a type with its own show method for this type of table
function coeftable(m::LinearMixedModel)
    fe = fixef(m); se = stderr(m)
    DataFrame({fe, se, fe./se}, ["Estimate","Std.Error","z value"])
end

## deviance(m) -> Float64
deviance(m::LinearMixedModel) = m.fit && !m.REML ? objective(m) : NaN
        
## fixef(m) -> current value of beta (can be a reference)
fixef(m::LinearMixedModel) = m.beta

## fnames(m) -> names of grouping factors
fnames(m::LinearMixedModel) = m.fnms

##  isfit(m) -> Bool - Has the model been fit?
isfit(m::LinearMixedModel) = m.fit

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
rss(m::LinearMixedModel) = sumsqdiff(m.mu, m.y)

## scale(m) -> estimate, s, of the scale parameter
## scale(m,true) -> estimate, s^2, of the squared scale parameter
function scale(m::LinearMixedModel, sqr=false)
    n,p = size(m); ssqr = pwrss(m)/float64(n - (m.REML ? p : 0))
    sqr ? ssqr : sqrt(ssqr)
end

function show(io::IO, m::LinearMixedModel)
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
    stdm = std(m); fnms = vcat(fnames(m),"Residual")
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
    tstrings = split(string(coeftable(m)),'\n')
    for i in 2:p+2 print(io,tstrings[i]); print(io,"\n") end
end

##  size(m) -> n, p, q, t (lengths of y, beta, u and # of re terms)
size(m::LinearMixedModel) = (length(m.y), length(m.beta),
                             sum([length(u) for u in m.u]), length(m.u))

## stderr(m) -> standard errors of fixed-effects parameters
stderr(m::LinearMixedModel) = sqrt(diag(vcov(m)))

## vcov(m) -> estimated variance-covariance matrix of the fixed-effects parameters
vcov(m::LinearMixedModel) = scale(m,true) * inv(cholfact(m))
