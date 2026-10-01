
"""
     profilevc(m::LinearMixedModel{T}, val::T, rowj::AbstractVector{T}) where {T}

Profile an element of the variance components.

!!! note
    This method is called by `profile` and currently considered internal.
    As such, it may change or disappear in a future release without being considered breaking.
"""
function profilevc(m::LinearMixedModel{T}, val::T, rowj::AbstractVector{T}) where {T}
    optsum = m.optsum
    # by giving g a default we can also
    # work with backends which don't have a gradient slot
    function obj(x, g=T[])
        isempty(g) || throw(ArgumentError("g must be empty"))
        updateL!(setθ!(m, x))
        optsum.sigma = val / norm(rowj)
        objctv = objective(m)
        return objctv
    end

    return profilevc(obj, optsum, Val(m.optsum.backend))
end

"""
    _objective_vczero!(m::LinearMixedModel{T}, θ̂::Vector{T}, t::Integer, k::Integer) where {T}

Return the minimum of the objective of `m` when the `k`th variance component of the `t`th
random-effects term is zero, leaving `m` at the minimizer.

The variance component is zero when the `k`th row of `λ` for the term is zero, so those
elements of θ are held at zero and the others, starting from `θ̂`, are optimized.

!!! note
    This method is internal.
"""
function _objective_vczero!(
    m::LinearMixedModel{T}, θ̂::Vector{T}, t::Integer, k::Integer
) where {T}
    (; optsum, parmap) = m
    θ = copy(θ̂)
    fixed = findall(pm -> pm[1] == t && pm[2] == k, parmap)
    θ[fixed] .= zero(T)
    free = setdiff(eachindex(θ), fixed)
    isempty(free) && return objective!(m, θ)
    osj = OptSummary(θ̂[free], optsum.optimizer; optsum.backend)
    function obj(x, g=T[])
        isempty(g) || throw(ArgumentError("gradients are not evaluated by this objective"))
        for (i, f) in enumerate(free)
            @inbounds θ[f] = x[i]
        end
        return objective!(m, θ)
    end
    return profileobj!(obj, m, θ, osj, Val(osj.backend))
end

"""
     profileσs!(val::NamedTuple, tc::TableColumns{T}; threshold=4) where {T}

Profile the variance components.

If the profile of a variance component has not reached `-threshold` at the lower end of its
grid, the point at which the variance component is zero is added.

!!! note
    This method is called by `profile` and currently considered internal.
    As such, it may change or disappear in a future release without being considered breaking.
"""
function profileσs!(val::NamedTuple, tc::TableColumns{T}; threshold=4) where {T}
    m = val.m
    (; optsum, reterms) = m
    isnothing(optsum.sigma) || throw(ArgumentError("Can't profile vc's when σ is fixed"))
    (; initial, final, fmin) = optsum
    saveinitial = copy(initial)
    θ̂ = copy(final)                          # profilevc overwrites final in place
    zetazero = mkrow!(tc, m, zero(T))         # parameter estimates
    vcnms = filter(keys(first(val.tbl))) do sym
        str = string(sym)
        return startswith(str, 'σ') && (length(str) > 1)
    end
    ind = 0
    for (ti, t) in enumerate(reterms)
        for (k, r) in enumerate(eachrow(t.λ))
            optsum.sigma = nothing            # re-initialize the model
            objective!(m, θ̂)
            copyto!(initial, θ̂)              # start each component from the estimates
            ind += 1
            sym = vcnms[ind]
            gpsym = getproperty(sym)          # extractor function
            estimate = gpsym(zetazero)
            pnm = (; p=sym)
            tbl = [merge(pnm, zetazero)]
            xtrms = extrema(gpsym, val.tbl)
            lub = log(last(xtrms))
            llb = log(max(first(xtrms), T(0.01) * last(xtrms)))
            # add the profile at x to tbl, starting the optimization from initial
            function vcpoint!(x)
                obj, xmin = profilevc(m, x, r)
                copyto!(initial, xmin)
                zeta = sign(x - estimate) * sqrt(max(zero(T), obj - fmin))
                push!(tbl, merge(pnm, mkrow!(tc, m, zeta)))
                return zeta
            end
            # start at the upper bound where things are more stable
            ζgrid = [vcpoint!(exp(lx)) for lx in LinRange(lub, llb, 15)]
            # Extend the grid with the same spacing on the log scale until the profile reaches
            # ±threshold. Going down, stop once the profile is flat, which leaves the rest to
            # the point at zero below, and do not go below 1% of the estimate, where the
            # optimization in profilevc becomes unreliable.
            δgrid = max((lub - llb) / 14, T(1//20))
            if !iszero(estimate)
                lx, ζlast, n = llb, ζgrid[end], 0
                Δζ = ζgrid[end - 1] - ζgrid[end]
                while ζlast > -threshold && Δζ ≥ T(1//10) &&
                          exp(lx - δgrid) > estimate / 100 && n < 30
                    ζnew = vcpoint!(exp(lx -= δgrid))
                    Δζ, ζlast, n = ζlast - ζnew, ζnew, n + 1
                end
            end
            # add the point at zero if the profile has not reached the threshold before it
            if !iszero(estimate) && minimum(getproperty(:ζ), tbl) > -threshold
                optsum.sigma = nothing
                obj = _objective_vczero!(m, θ̂, ti, k)
                push!(tbl, merge(pnm, mkrow!(tc, m, _ζ(obj, fmin, true, sym, zero(T)))))
            end
            copyto!(initial, θ̂)
            lx, ζlast, n = lub, first(ζgrid), 0
            while ζlast < threshold && n < 30
                ζlast = vcpoint!(exp(lx += δgrid))
                n += 1
            end
            sort!(tbl; by=gpsym)
            append!(val.tbl, tbl)
            ζcol = getproperty(:ζ).(tbl)
            symcol = gpsym.(tbl)
            try
                val.fwd[sym] = interpolate(symcol, ζcol, BSplineOrder(4), Natural())
                issorted(ζcol) &&
                    (val.rev[sym] = interpolate(ζcol, symcol, BSplineOrder(4), Natural()))
            catch
                @error "An error occurred while fitting the profile splines for the variance components. Try adjusting the threshold."
                rethrow()
            end
        end
    end
    copyto!(final, θ̂)
    copyto!(initial, saveinitial)
    optsum.sigma = nothing
    updateL!(setθ!(m, θ̂))
    return val
end
