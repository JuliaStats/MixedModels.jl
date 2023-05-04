
"""
     profilevc(m::LinearMixedModel{T}, val::T, rowj::AbstractVector{T}) where {T}

Profile an element of the variance components.

!!! note
    This method is called by `profile` and currently considered internal.
    As such, it may change or disappear in a future release without being considered breaking.
"""
function profilevc(m::LinearMixedModel{T}, val::T, rowj::AbstractVector{T}) where {T}
    optsum = m.optsum
    function obj(x, g)
        isempty(g) || throw(ArgumentError("g must be empty"))
        updateL!(setθ!(m, x))
        optsum.sigma = val / norm(rowj)
        objctv = objective(m)
        return objctv
    end
    opt = Opt(optsum)
    NLopt.min_objective!(opt, obj)
    fmin, xmin, ret = NLopt.optimize!(opt, copyto!(optsum.final, optsum.initial))
    _check_nlopt_return(ret)
    return fmin, xmin
end

"""
     profileσs!(val::NamedTuple, tc::TableColumns{T}; nzlb=1.0e-8) where {T}

Profile the variance components.

!!! note
    This method is called by `profile` and currently considered internal.
    As such, it may change or disappear in a future release without being considered breaking.
"""
function profileσs!(val::NamedTuple, tc::TableColumns{T}; nzlb=1.0e-8) where {T}
    m = val.m
    (; λ, σ, β, optsum, parmap, reterms) = m
    isnothing(optsum.sigma) || throw(ArgumentError("Can't profile vc's when σ is fixed"))
    (; initial, final, fmin, lowerbd) = optsum
    lowerbd .+= T(nzlb)                       # lower bounds must be > 0 b/c θ's occur in denominators
    saveinitial = copy(initial)
    copyto!(initial, max.(final, lowerbd))
    zetazero = mkrow!(tc, m, zero(T))         # parameter estimates
    vcnms = filter(keys(first(val.tbl))) do sym
        str = string(sym)
        return startswith(str, 'σ') && (length(str) > 1)
    end
    ind = 0
    for t in reterms
        for r in eachrow(t.λ)
            optsum.sigma = nothing            # re-initialize the model
            objective!(m, final)
            ind += 1
            sym = vcnms[ind]
            gpsym = getproperty(sym)          # extractor function
            estimate = gpsym(zetazero)
            pnm = (; p=sym)
            tbl = [merge(pnm, zetazero)]
            xtrms = extrema(gpsym, val.tbl)
            lub = log(last(xtrms))
            llb = log(max(first(xtrms), T(0.01) * lub))
            for lx in LinRange(lub, llb, 15)  # start at the upper bound where things are more stable
                x = exp(lx)
                obj, xmin = profilevc(m, x, r)
                copyto!(initial, xmin)
                zeta = sign(x - estimate) * sqrt(max(zero(T), obj - fmin))
                push!(tbl, merge(pnm, mkrow!(tc, m, zeta)))
            end
            if iszero(first(xtrms)) && !iszero(estimate) # handle the case of lower bound of zero
                zrows = filter(iszero ∘ gpsym, val.tbl)
                isone(length(zrows)) ||
                    filter!(r -> iszero(getproperty(r, first(r))), zrows)
                rr = only(zrows)              # will error if zeros in sym column occur in unexpected places
                push!(tbl, merge(pnm, rr[(collect(keys(rr))[2:end]...,)]))
            end
            sort!(tbl; by=gpsym)
            append!(val.tbl, tbl)
            ζcol = getproperty(:ζ).(tbl)
            symcol = gpsym.(tbl)
            val.fwd[sym] = interpolate(symcol, ζcol, BSplineOrder(4), Natural())
            issorted(ζcol) &&
                (val.rev[sym] = interpolate(ζcol, symcol, BSplineOrder(4), Natural()))
        end
    end
    copyto!(final, initial)
    copyto!(initial, saveinitial)
    lowerbd .-= T(nzlb)
    optsum.sigma = nothing
    updateL!(setθ!(m, final))
    return val
end
