
function profilevc(m::LinearMixedModel{T}, val::T, rowj::AbstractVector{T}) where {T}
    optsum = m.optsum
    function obj(x, g)
        isempty(g) || throw(ArgumentError("g must be empty"))
        updateL!(setθ!(m, x))
        optsum.sigma = val / norm(rowj) 
        return objective(m)
    end
    opt = Opt(optsum)
    NLopt.min_objective!(opt, obj)
    fmin, xmin, ret = NLopt.optimize!(opt, copyto!(optsum.final, optsum.initial))
    _check_nlopt_return(ret)
    return fmin
end

function profileσs!(pr::MixedModelProfile{T}, tc::TableColumns{T}) where {T}
    m = pr.m
    @compat (; λ, σ, β, optsum, parmap, reterms) = m
    isnothing(optsum.sigma) || throw(ArgumentError("Can't profile vc's when σ is fixed"))
    @compat (; initial, final, fmin) = optsum
    saveinitial = copy(initial)
    copyto!(initial, final)
    zetazero = mkrow!(tc, m, zero(T)) # parameter estimates
    vcnms = filter(sym -> (str = string(sym); startswith(str, 'σ') && length(str) > 1), keys(first(pr.tbl)))
    ind = 0
    for t in reterms
        for r in eachrow(t.λ.data)
            optsum.sigma = nothing        # re-initialize the model
            objective!(m, final)
            ind += 1
            sym = vcnms[ind]
            gpsym = getproperty(sym)      # extractor function
            estimate = gpsym(zetazero)
            pnm = (; p = sym,)
            tbl = [merge(pnm, zetazero)]
            xtrms = extrema(gpsym, pr.tbl)
            lub = log(last(xtrms))
            llb = log(max(first(xtrms), T(0.01) * lub))
            for lx in LinRange(llb, lub, 15)
                x = exp(lx)
                obj = profilevc(m, x, r)
                zeta = sign(x - estimate) * sqrt(max(zero(T), obj - fmin))
                push!(tbl, merge(pnm, mkrow!(tc, m, zeta)))
            end
            if iszero(first(xtrms)) && !iszero(estimate)      # handle the case of lower bound of zero
                zrows = filter(iszero ∘ gpsym, pr.tbl)
                isone(length(zrows)) || filter!(r -> iszero(getproperty(r, first(r))), zrows)
                rr = only(zrows)          # will error if zeros in sym column occur in unexpected places
                push!(tbl, merge(pnm, rr[(collect(keys(rr))[2:end]...,)]))
            end
            sort!(tbl; by = gpsym)
            append!(pr.tbl, tbl)
            ζcol = getproperty(:ζ).(tbl)
            symcol = gpsym.(tbl)
            pr.fwd[sym] = interpolate(symcol, ζcol, BSplineOrder(4), Natural())
            issorted(ζcol) && (pr.rev[sym] = interpolate(ζcol, symcol, BSplineOrder(4), Natural()))
        end
    end
    copyto!(final, initial)
    copyto!(initial, saveinitial)
    optsum.sigma = nothing
    updateL!(setθ!(m, final))
    return pr
end
