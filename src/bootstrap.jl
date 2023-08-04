"""
    MixedModelFitCollection{T<:AbstractFloat}

Abstract supertype for [`MixedModelBootstrap`](@ref) and related functionality in other packages.
"""
abstract type MixedModelFitCollection{T<:AbstractFloat} end

"""
    MixedModelBootstrap{T<:AbstractFloat} <: MixedModelFitCollection{T}

Object returned by `parametericbootstrap` with fields
- `fits`: the parameter estimates from the bootstrap replicates as a vector of named tuples.
- `λ`: `Vector{LowerTriangular{T,Matrix{T}}}` containing copies of the λ field from `ReMat` model terms
- `inds`: `Vector{Vector{Int}}` containing copies of the `inds` field from `ReMat` model terms
- `lowerbd`: `Vector{T}` containing the vector of lower bounds (corresponds to the identically named field of [`OptSummary`](@ref))
- `fcnames`: NamedTuple whose keys are the grouping factor names and whose values are the column names

The schema of `fits` is, by default,
```
Tables.Schema:
 :objective  T
 :σ          T
 :β          NamedTuple{β_names}{NTuple{p,T}}
 :se         StaticArrays.SArray{Tuple{p},T,1,p}
 :θ          StaticArrays.SArray{Tuple{k},T,1,k}
```
where the sizes, `p` and `k`, of the `β` and `θ` elements are determined by the model.

Characteristics of the bootstrap replicates can be extracted as properties.  The `σs` and
`σρs` properties unravel the `σ` and `θ` estimates into estimates of the standard deviations
and correlations of the random-effects terms.
"""
struct MixedModelBootstrap{T<:AbstractFloat} <: MixedModelFitCollection{T}
    fits::Vector
    λ::Vector{Union{LowerTriangular{T},Diagonal{T}}}
    inds::Vector{Vector{Int}}
    lowerbd::Vector{T}
    fcnames::NamedTuple
end

Base.:(==)(a::MixedModelFitCollection{T}, b::MixedModelFitCollection{S}) where {T,S} = false

function Base.:(==)(a::MixedModelFitCollection{T}, b::MixedModelFitCollection{T}) where {T}
    return a.fits == b.fits &&
           a.λ == b.λ &&
           a.inds == b.inds &&
           a.lowerbd == b.lowerbd &&
           a.fcnames == b.fcnames
end

function Base.isapprox(a::MixedModelFitCollection, b::MixedModelFitCollection;
    atol::Real=0, rtol::Real=atol > 0 ? 0 : √eps())
    fits = all(zip(a.fits, b.fits)) do (x, y)
        return isapprox(x.objective, y.objective; atol, rtol) &&
               isapprox(x.θ, y.θ; atol, rtol) &&
               isapprox(x.σ, y.σ; atol, rtol) &&
               all(isapprox(a, b; atol, rtol) for (a, b) in zip(x.β, y.β))
    end

    λ = all(zip(a.λ, b.λ)) do (x, y)
        return isapprox(x, y; atol, rtol)
    end

    return fits && λ &&
           # Vector{Vector{Int}} so no need for isapprox
           a.inds == b.inds &&
           isapprox(a.lowerbd, b.lowerbd; atol, rtol) &&
           a.fcnames == b.fcnames
end

"""
    restorereplicates(f, m::MixedModel{T})
    restorereplicates(f, m::MixedModel{T}, ftype::Type{<:AbstractFloat})
    restorereplicates(f, m::MixedModel{T}, ctype::Type{<:MixedModelFitCollection{S}})

Restore replicates from `f`, using `m` to create the desired subtype of [`MixedModelFitCollection`](@ref).

`f` can be any entity suppored by `Arrow.Table`. `m` does not have to be fitted, but it must have
been constructed with the same structure as the source of the saved replicates.

The two-argument method constructs a [`MixedModelBootstrap`](@ref) with the same eltype as `m`.
If an eltype is specified as the third argument, then a `MixedModelBootstrap` is returned.
If a subtype of `MixedModelFitCollection` is specified as the third argument, then that 
is the return type.

See also [`savereplicates`](@ref), [`restoreoptsum!`](@ref).
"""
function restorereplicates(f, m::MixedModel{T}, ftype::Type{<:AbstractFloat}=T) where {T}
    return restorereplicates(f, m, MixedModelBootstrap{ftype})
end

# why this weird second method? it allows us to define custom types and write methods
# to load into those types directly. For example, we could define a `PowerAnalysis <: MixedModelFitCollection`
# in MixedModelsSim and then overload this method to get a convenient object. 
# Also, this allows us to write `restorereplicateS(f, m, ::Type{<:MixedModelNonparametricBoostrap})` for
# entities in MixedModels bootstrap
function restorereplicates(
    f, m::MixedModel, ctype::Type{<:MixedModelFitCollection{T}}
) where {T}
    tbl = Arrow.Table(f)
    # use a lazy iterator to get the first element for checks
    # before doing a conversion of the entire Arrow column table to row table
    rep = first(Tables.rows(tbl))
    allgood =
        length(rep.θ) == length(m.θ) &&
        string.(propertynames(rep.β)) == Tuple(coefnames(m))
    allgood ||
        throw(ArgumentError("Model is not compatible with saved replicates."))

    samp = Tables.rowtable(tbl)
    return ctype(
        samp,
        map(vv -> T.(vv), m.λ), # also does a deepcopy if no type conversion is necessary
        getfield.(m.reterms, :inds),
        T.(m.optsum.lowerbd[1:length(first(samp).θ)]),
        NamedTuple{Symbol.(fnames(m))}(map(t -> Tuple(t.cnames), m.reterms)),
    )
end

"""
    savereplicates(f, b::MixedModelFitCollection)

Save the replicates associated with a [`MixedModelFitCollection`](@ref), 
e.g. [`MixedModelBootstrap`](@ref) as an Arrow file. 

See also [`restorereplicates`](@ref), [`saveoptsum`](@ref)

!!! note
    **Only** the replicates are saved, not the entire contents of the `MixedModelFitCollection`.
    `restorereplicates` requires a model compatible with the bootstrap to restore the full object. 
"""
savereplicates(f, b::MixedModelFitCollection) = Arrow.write(f, b.fits)

# TODO: write methods for GLMM
function Base.vcat(b1::MixedModelBootstrap{T}, b2::MixedModelBootstrap{T}) where {T}
    for field in [:λ, :inds, :lowerbd, :fcnames]
        getfield(b1, field) == getfield(b2, field) ||
            throw(ArgumentError("b1 and b2 must originate from the same model fit"))
    end
    return MixedModelBootstrap{T}(vcat(b1.fits, b2.fits),
        deepcopy(b1.λ),
        deepcopy(b1.inds),
        deepcopy(b1.lowerbd),
        deepcopy(b1.fcnames))
end

function Base.reduce(::typeof(vcat), v::AbstractVector{MixedModelBootstrap{T}}) where {T}
    for field in [:λ, :inds, :lowerbd, :fcnames]
        all(==(getfield(first(v), field)), getfield.(v, field)) ||
            throw(ArgumentError("All bootstraps must originate from the same model fit"))
    end

    b1 = first(v)
    fits = reduce(vcat, getfield.(v, :fits))
    return MixedModelBootstrap{T}(fits,
        deepcopy(b1.λ),
        deepcopy(b1.inds),
        deepcopy(b1.lowerbd),
        deepcopy(b1.fcnames))
end

"""
    parametricbootstrap([rng::AbstractRNG], nsamp::Integer, m::MixedModel{T}, ftype=T;
        β = coef(m), σ = m.σ, θ = m.θ, hide_progress=false, optsum_overrides=(;))

Perform `nsamp` parametric bootstrap replication fits of `m`, returning a `MixedModelBootstrap`.

The default random number generator is `Random.GLOBAL_RNG`.

`ftype` can be used to store the computed bootstrap values in a lower precision. `ftype` is
not a named argument because named arguments are not used in method dispatch and thus
specialization. In other words, having `ftype` as a positional argument has some potential
performance benefits.

# Keyword Arguments

- `β`, `σ`, and `θ` are the values of `m`'s parameters for simulating the responses.
- `σ` is only valid for `LinearMixedModel` and `GeneralizedLinearMixedModel` for
families with a dispersion parameter.
- `hide_progress` can be used to disable the progress bar. Note that the progress
bar is automatically disabled for non-interactive (i.e. logging) contexts.
- `optsum_overrides` is used to override values of [OptSummary](@ref) in the models
fit during the bootstrapping process. For example, `optsum_overrides=(;ftol_rel=1e08)`
reduces the convergence criterion, which can greatly speed up the bootstrap fits.
Taking advantage of this speed up to increase `n` can often lead to better estimates
of coverage intervals.
"""
function parametricbootstrap(
    rng::AbstractRNG,
    n::Integer,
    morig::MixedModel{T},
    ftype::Type{<:AbstractFloat}=T;
    β::AbstractVector=coef(morig),
    σ=morig.σ,
    θ::AbstractVector=morig.θ,
    use_threads::Bool=false,
    hide_progress::Bool=false,
    optsum_overrides=(;),
) where {T}
    if σ !== missing
        σ = T(σ)
    end
    β, θ = convert(Vector{T}, β), convert(Vector{T}, θ)
    βsc, θsc = similar(ftype.(β)), similar(ftype.(θ))
    p, k = length(β), length(θ)
    m = deepcopy(morig)
    for (key, val) in pairs(optsum_overrides)
        setfield!(m.optsum, key, val)
    end
    # this seemed to slow things down?!
    # _copy_away_from_lowerbd!(m.optsum.initial, morig.optsum.final, m.lowerbd; incr=0.05)

    β_names = Tuple(Symbol.(fixefnames(morig)))

    use_threads && Base.depwarn(
        "use_threads is deprecated and will be removed in a future release",
        :parametricbootstrap,
    )
    samp = replicate(n; hide_progress=hide_progress) do
        simulate!(rng, m; β, σ, θ)
        refit!(m; progress=false)
        # @info "" m.optsum.feval
        (
            objective=ftype.(m.objective),
            σ=ismissing(m.σ) ? missing : ftype(m.σ),
            β=NamedTuple{β_names}(fixef!(βsc, m)),
            se=SVector{p,ftype}(stderror!(βsc, m)),
            θ=SVector{k,ftype}(getθ!(θsc, m)),
        )
    end
    return MixedModelBootstrap{ftype}(
        samp,
        map(vv -> ftype.(vv), morig.λ), # also does a deepcopy if no type conversion is necessary
        getfield.(morig.reterms, :inds),
        ftype.(morig.optsum.lowerbd[1:length(first(samp).θ)]),
        NamedTuple{Symbol.(fnames(morig))}(map(t -> Tuple(t.cnames), morig.reterms)),
    )
end

function parametricbootstrap(nsamp::Integer, m::MixedModel, args...; kwargs...)
    return parametricbootstrap(Random.GLOBAL_RNG, nsamp, m, args...; kwargs...)
end

"""
    allpars(bsamp::MixedModelFitCollection)

Return a tidy (column)table with the parameter estimates spread into columns
of `iter`, `type`, `group`, `name` and `value`.

!!! warning
    Currently, correlations that are systematically zero are included in the
    the result. This may change in a future release without being considered
    a breaking change.
"""
function allpars(bsamp::MixedModelFitCollection{T}) where {T}
    fits, λ, fcnames = bsamp.fits, bsamp.λ, bsamp.fcnames
    npars = 2 + length(first(fits).β) + sum(map(k -> (k * (k + 1)) >> 1, size.(bsamp.λ, 2)))
    nresrow = length(fits) * npars
    cols = (
        sizehint!(Int[], nresrow),
        sizehint!(String[], nresrow),
        sizehint!(Union{Missing,String}[], nresrow),
        sizehint!(Union{Missing,String}[], nresrow),
        sizehint!(T[], nresrow),
    )
    nrmdr = Vector{T}[]  # normalized rows of λ
    for (i, r) in enumerate(fits)
        σ = coalesce(r.σ, one(T))
        for (nm, v) in pairs(r.β)
            push!.(cols, (i, "β", missing, String(nm), v))
        end
        setθ!(bsamp, i)
        for (grp, ll) in zip(keys(fcnames), λ)
            rownms = getproperty(fcnames, grp)
            grpstr = String(grp)
            empty!(nrmdr)
            for (j, rnm, row) in zip(eachindex(rownms), rownms, eachrow(ll))
                push!.(cols, (i, "σ", grpstr, rnm, σ * norm(row)))
                push!(nrmdr, normalize(row))
                for k in 1:(j - 1)
                    push!.(
                        cols,
                        (
                            i,
                            "ρ",
                            grpstr,
                            string(rownms[k], ", ", rnm),
                            dot(nrmdr[j], nrmdr[k]),
                        ),
                    )
                end
            end
        end
        r.σ === missing || push!.(cols, (i, "σ", "residual", missing, r.σ))
    end
    return (
        iter=cols[1],
        type=PooledArray(cols[2]),
        group=PooledArray(cols[3]),
        names=PooledArray(cols[4]),
        value=cols[5],
    )
end

"""
    confint(pr::MixedModelBootstrap; level::Real=0.95)

Compute bootstrap confidence intervals for coefficients and variance components, with confidence level level (by default 95%).

!!! note
    The API guarantee is for a Tables.jl compatible table. The exact return type is an
    implementation detail and may change in a future minor release without being considered
    breaking.

!!! note
    The "row names" indicating the associated parameter name are guaranteed to be unambiguous,
    but their precise naming scheme is not yet stable and may change in a future release
    without being considered breaking.

See also [`shortestcovint`](@ref).
"""
function StatsBase.confint(bsamp::MixedModelBootstrap{T}; level::Real=0.95) where {T}
    cutoff = sqrt(quantile(Chisq(1), level))
    # Creating the table is somewhat wasteful because columns are created then immediately skipped.
    tbl = Table(bsamp.tbl)
    lower = T[]
    upper = T[]
    v = similar(tbl.σ)
    par = sort!(
        collect(
            filter(
                k -> !(startswith(string(k), 'θ') || string(k) == "obj"), propertynames(tbl)
            ),
        ),
    )
    for p in par
        l, u = shortestcovint(sort!(copyto!(v, getproperty(tbl, p))), level)
        push!(lower, l)
        push!(upper, u)
    end
    return DictTable(; par, lower, upper)
end

function Base.getproperty(bsamp::MixedModelFitCollection, s::Symbol)
    if s ∈ [:objective, :σ, :θ, :se]
        getproperty.(getfield(bsamp, :fits), s)
    elseif s == :β
        tidyβ(bsamp)
    elseif s == :coefpvalues
        coefpvalues(bsamp)
    elseif s == :σs
        tidyσs(bsamp)
    elseif s == :allpars
        allpars(bsamp)
    elseif s == :tbl
        pbstrtbl(bsamp)
    else
        getfield(bsamp, s)
    end
end

"""
    issingular(bsamp::MixedModelFitCollection)

Test each bootstrap sample for singularity of the corresponding fit.

Equality comparisons are used b/c small non-negative θ values are replaced by 0 in `fit!`.

See also [`issingular(::MixedModel)`](@ref).
"""
issingular(bsamp::MixedModelFitCollection) = map(θ -> any(θ .== bsamp.lowerbd), bsamp.θ)

Base.length(x::MixedModelFitCollection) = length(x.fits)

function Base.propertynames(bsamp::MixedModelFitCollection)
    return [
        :allpars,
        :objective,
        :σ,
        :β,
        :se,
        :coefpvalues,
        :θ,
        :σs,
        :λ,
        :inds,
        :lowerbd,
        :fits,
        :fcnames,
        :tbl,
    ]
end

"""
    setθ!(bsamp::MixedModelFitCollection, i::Integer)

Install the values of the i'th θ value of `bsamp.fits` in `bsamp.λ`
"""
function setθ!(bsamp::MixedModelFitCollection, i::Integer)
    θ = bsamp.fits[i].θ
    offset = 0
    for (λ, inds) in zip(bsamp.λ, bsamp.inds)
        λdat = _getdata(λ)
        fill!(λdat, false)
        for j in eachindex(inds)
            λdat[inds[j]] = θ[j + offset]
        end
        offset += length(inds)
    end
    return bsamp
end

_getdata(x::Diagonal) = x
_getdata(x::LowerTriangular) = x.data

"""
    shortestcovint(v, level = 0.95)

Return the shortest interval containing `level` proportion of the values of `v`
"""
function shortestcovint(v, level=0.95)
    n = length(v)
    0 < level < 1 || throw(ArgumentError("level = $level should be in (0,1)"))
    vv = issorted(v) ? v : sort(v)
    ilen = Int(ceil(n * level)) # number of elements (counting endpoints) in interval
    # skip non-finite elements at the ends of sorted vv
    start = findfirst(isfinite, vv)
    stop = findlast(isfinite, vv)
    if stop < (start + ilen - 1)
        return (vv[1], vv[end])
    end
    len, i = findmin([vv[i + ilen - 1] - vv[i] for i in start:(stop + 1 - ilen)])
    return (vv[i], vv[i + ilen - 1])
end

"""
    shortestcovint(bsamp::MixedModelFitCollection, level = 0.95)

Return the shortest interval containing `level` proportion for each parameter from `bsamp.allpars`.

!!! warning
    Currently, correlations that are systematically zero are included in the
    the result. This may change in a future release without being considered
    a breaking change.
"""
function shortestcovint(bsamp::MixedModelFitCollection{T}, level=0.95) where {T}
    allpars = bsamp.allpars
    pars = unique(zip(allpars.type, allpars.group, allpars.names))

    colnms = (:type, :group, :names, :lower, :upper)
    coltypes = Tuple{String,Union{Missing,String},Union{Missing,String},T,T}
    # not specifying the full eltype (NamedTuple{colnms,coltypes}) leads to prettier printing
    result = NamedTuple{colnms}[]
    sizehint!(result, length(pars))

    for (t, g, n) in pars
        gidx = if ismissing(g)
            ismissing.(allpars.group)
        else
            .!ismissing.(allpars.group) .& (allpars.group .== g)
        end

        nidx = if ismissing(n)
            ismissing.(allpars.names)
        else
            .!ismissing.(allpars.names) .& (allpars.names .== n)
        end

        tidx = allpars.type .== t # no missings allowed here

        idx = tidx .& gidx .& nidx

        vv = view(allpars.value, idx)

        lower, upper = shortestcovint(vv, level)
        push!(result, (; type=t, group=g, names=n, lower=lower, upper=upper))
    end

    return result
end

"""
    tidyβ(bsamp::MixedModelFitCollection)
Return a tidy (row)table with the parameter estimates spread into columns
of `iter`, `coefname` and `β`
"""
function tidyβ(bsamp::MixedModelFitCollection{T}) where {T}
    fits = bsamp.fits
    colnms = (:iter, :coefname, :β)
    result = sizehint!(
        NamedTuple{colnms,Tuple{Int,Symbol,T}}[], length(fits) * length(first(fits).β)
    )
    for (i, r) in enumerate(fits)
        for (k, v) in pairs(r.β)
            push!(result, NamedTuple{colnms}((i, k, v)))
        end
    end
    return result
end

"""
    coefpvalues(bsamp::MixedModelFitCollection)

Return a rowtable with columns `(:iter, :coefname, :β, :se, :z, :p)`
"""
function coefpvalues(bsamp::MixedModelFitCollection{T}) where {T}
    fits = bsamp.fits
    colnms = (:iter, :coefname, :β, :se, :z, :p)
    result = sizehint!(
        NamedTuple{colnms,Tuple{Int,Symbol,T,T,T,T}}[], length(fits) * length(first(fits).β)
    )
    for (i, r) in enumerate(fits)
        for (p, s) in zip(pairs(r.β), r.se)
            β = last(p)
            z = β / s
            push!(result, NamedTuple{colnms}((i, first(p), β, s, z, 2normccdf(abs(z)))))
        end
    end
    return result
end

"""
    tidyσs(bsamp::MixedModelFitCollection)

Return a tidy (row)table with the estimates of the variance components (on the standard deviation scale) spread into columns
of `iter`, `group`, `column` and `σ`.
"""
function tidyσs(bsamp::MixedModelFitCollection{T}) where {T}
    fits = bsamp.fits
    fcnames = bsamp.fcnames
    λ = bsamp.λ
    colnms = (:iter, :group, :column, :σ)
    result = sizehint!(
        NamedTuple{colnms,Tuple{Int,Symbol,Symbol,T}}[], length(fits) * sum(length, fcnames)
    )
    for (iter, r) in enumerate(fits)
        setθ!(bsamp, iter)    # install r.θ in λ
        σ = coalesce(r.σ, one(T))
        for (grp, ll) in zip(keys(fcnames), λ)
            for (cn, col) in zip(getproperty(fcnames, grp), eachrow(ll))
                push!(result, NamedTuple{colnms}((iter, grp, Symbol(cn), σ * norm(col))))
            end
        end
    end
    return result
end

_size1(m) = (first ∘ size)(m)

_nrho(m) = (kchoose2 ∘ _size1)(m)

function _appendsym!(
    syms::AbstractVector{Symbol},
    dict::Dict{Symbol,UnitRange{Int}},
    sym::Symbol,
    len::Integer,
)
    lenp1 = length(syms) + 1
    append!(syms, _generatesyms(first(string(sym)), len))
    dict[sym] = lenp1:(length(syms))
    return syms
end

function _prototypevec(bsamp::MixedModelBootstrap)
    (; fits, λ) = bsamp
    (; β, θ) = first(fits)
    dict = Dict{Symbol,UnitRange{Int}}(:obj => 1:1, :σ => 2:2)
    syms = [:obj, :σ]
    _appendsym!(syms, dict, :β, length(β))
    _appendsym!(syms, dict, :σs, sum(_size1, λ))
    _appendsym!(syms, dict, :ρ, sum(_nrho, λ))
    _appendsym!(syms, dict, :θ, length(θ))
    return Tuple(syms), dict
end

function _allpars!(
    v::AbstractVector{T},
    bsamp::MixedModelBootstrap{T},
    i::Integer,
    d::Dict{Symbol,UnitRange{Int}},
) where {T}
    fiti = bsamp.fits[i]
    λ = bsamp.λ
    v[1] = fiti.objective
    v[2] = σ = coalesce(fiti.σ, one(T))
    copyto!(view(v, d[:β]), fiti.β)
    copyto!(view(v, d[:θ]), fiti.θ)
    k = first(d[:σs])
    setθ!(bsamp, i)
    if isa(λ, Diagonal)
        for d in λ.diag
            v[k] = σ * d
            k += 1
        end
    elseif isa(λ, LowerTriangular)
        for l in λ
            for λr in eachrow(l.data)
                v[k] = σ * norm(λr)
                k += 1
            end
        end
    end
    drho = d[:ρ]
    if !isempty(drho)
        fill!(view(v, drho), zero(T))
        k = first(drho)
        for ll in λ
            if isa(ll, LowerTriangular{T})
                lam = ll.data
                ii = _size1(lam)
                isone(ii) && continue
                for i in 1:ii
                    ri = normalize!(view(lam, i, :))
                    for j in 1:(i - 1)
                        v[k] = dot(ri, view(lam, j, :))
                        k += 1
                    end
                end
            end
        end
    end
    return v
end

function pbstrtbl(bsamp::MixedModelFitCollection{T}) where {T}
    (; fits, λ) = bsamp
    syms, d = _prototypevec(bsamp)
    nsym = length(syms)
    val = NamedTuple{syms,NTuple{nsym,T}}[]
    v = Vector{T}(undef, nsym)
    for i in axes(fits, 1)
        push!(val, NamedTuple{syms,NTuple{nsym,T}}(_allpars!(v, bsamp, i, d)))
    end
    return val
end

"""
    _rowlengths(v::AbstractVector, λ::Diagonal)

Copy the row lengths (absolute value of the diagonal elements) of `λ`` into `v`
"""
function _rowlengths(v::AbstractVector{T}, λ::Diagonal{T}) where {T}
    v .= abs.(λ.diag)
    return v
end

function _rowlengths(v::AbstractVector{T}, λ::LowerTriangular{T}) where {T}
    dat = λ.data
    k = size(dat, 1)
    fill!(v, zero(T))
    v[1] = abs(first(dat))
    for i in 2:k
        accum = zero(T)
        for j in 1:i
            accum += abs2(dat[i, j])
        end
        v[i] = sqrt(accum)
    end
    return v
end

"""
    _rowdotprods(v::AbstractVector{T}, λ)

Evaluate the dot products of rows of `λ` into v.  The dot products are stored in column-major order of the upper triangle.
"""
function _rowdotprods(v::AbstractVector{T}, λ::Diagonal{T}) where {T}
    fill!(v, zero(T))
    return v
end

function _rowdotprods(v::AbstractVector{T}, λ::LowerTriangular{T}) where {T}
    fill!(v, zero(T))
    dat = λ.data
    k = size(dat, 1)
    l = 1
    for i in 2:k
        for j in 1:(i - 1)
            accum = zero(T)
            for ii in 1:j
                accum += dat[i, ii] * dat[j, ii]
            end
            v[l] = accum
            l += 1
        end
    end
    return v
end
