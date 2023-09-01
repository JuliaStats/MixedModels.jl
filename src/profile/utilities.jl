"""
    TableColumns

A structure containing the column names for the numeric part of the profile table.

The struct also contains a Dict giving the column ranges for Symbols like `:σ` and `:β`.
Finally it contains a scratch vector used to accumulate to values in a row of the profile table.

!!! note
    This is an internal structure used in [`MixedModelProfile`](@ref).
    As such, it may change or disappear in a future release without being considered breaking.
"""
struct TableColumns{T<:AbstractFloat,N}
    cnames::NTuple{N,Symbol}
    positions::Dict{Symbol,UnitRange{Int}}
    v::Vector{T}
    corrpos::Vector{NTuple{3,Int}}
end

"""
    _generatesyms(tag::Char, len::Integer)

Utility to generate a vector of Symbols of the form :<tag><index> from a tag and a length.

The indices are left-padded with zeros to allow lexicographic sorting.
"""
function _generatesyms(tag::AbstractString, len::Integer)
    return Symbol.(string.(tag, lpad.(Base.OneTo(len), ndigits(len), '0')))
end

_generatesyms(tag::Char, len::Integer) = _generatesyms(string(tag), len)

function TableColumns(m::LinearMixedModel{T}) where {T}
    nmvec = [:ζ]
    positions = Dict(:ζ => 1:1)
    lastpos = 1
    sz = m.feterm.rank
    append!(nmvec, _generatesyms('β', sz))
    positions[:β] = (lastpos + 1):(lastpos + sz)
    lastpos += sz
    push!(nmvec, :σ)
    lastpos += 1
    positions[:σ] = lastpos:lastpos
    sz = sum(t -> size(t.λ, 1), m.reterms)
    append!(nmvec, _generatesyms('σ', sz))
    positions[:σs] = (lastpos + 1):(lastpos + sz)
    lastpos += sz
    corrpos = NTuple{3,Int}[]
    for (i, re) in enumerate(m.reterms)
        (isa(re.λ, Diagonal) || isa(re, ReMat{T,1})) && continue
        indm = indmat(re)
        for j in axes(indm, 1)
            rowj = view(indm, j, :)
            for k in (j + 1):size(indm, 1)
                if !iszero(dot(rowj, view(indm, k, :)))
                    push!(corrpos, (i, j, k))
                end
            end
        end
    end
    sz = length(corrpos)
    if sz > 0
        append!(nmvec, _generatesyms('ρ', sz))
        positions[:ρs] = (lastpos + 1):(lastpos + sz)
        lastpos += sz
    end
    sz = length(m.θ)
    append!(nmvec, _generatesyms('θ', sz))
    positions[:θ] = (lastpos + 1):(lastpos + sz)
    return TableColumns((nmvec...,), positions, zeros(T, length(nmvec)), corrpos)
end

function mkrow!(tc::TableColumns{T,N}, m::LinearMixedModel{T}, ζ::T) where {T,N}
    (; cnames, positions, v, corrpos) = tc
    v[1] = ζ
    fixef!(view(v, positions[:β]), m)
    v[first(positions[:σ])] = m.σ
    σvals!(view(v, positions[:σs]), m)
    getθ!(view(v, positions[:θ]), m)  # must do this first to preserve a copy
    if length(corrpos) > 0
        ρvals!(view(v, positions[:ρs]), corrpos, m)
        setθ!(m, view(v, positions[:θ]))
    end
    return NamedTuple{cnames,NTuple{N,T}}((v...,))
end

"""
    parsej(sym::Symbol)

Return the index from symbol names like `:θ1`, `:θ01`, etc.

!!! note
    This method is internal.
"""
function parsej(sym::Symbol)
    symstr = string(sym)                                     # convert Symbol to a String
    return parse(Int, SubString(symstr, nextind(symstr, 1))) # drop first Unicode character and parse as Int
end

#=  # It appears that this method is not used
"""
    σvals(m::LinearMixedModel)

Return a Tuple of the standard deviation estimates of the random effects
"""
function σvals(m::LinearMixedModel{T}) where {T}
    (; σ, reterms) = m
    isone(length(reterms)) && return σvals(only(reterms), σ)
    return (collect(Iterators.flatten(σvals.(reterms, σ)))...,)
end
=#

function σvals!(v::AbstractVector{T}, m::LinearMixedModel{T}) where {T}
    (; σ, reterms) = m
    isone(length(reterms)) && return σvals!(v, only(reterms), σ)
    ind = firstindex(v)
    for t in m.reterms
        S = size(t.λ, 1)
        σvals!(view(v, ind:(ind + S - 1)), t, σ)
        ind += S
    end
    return v
end

function ρvals!(
    v::AbstractVector{T}, corrpos::Vector{NTuple{3,Int}}, m::LinearMixedModel{T}
) where {T}
    reterms = m.reterms
    lasti = 1
    λ = first(reterms).λ
    for r in eachrow(λ)
        normalize!(r)
    end
    for (ii, pos) in enumerate(corrpos)
        i, j, k = pos
        if lasti ≠ i
            λ = reterms[i].λ
            for r in eachrow(λ)
                normalize!(r)
            end
            lasti = i
        end
        v[ii] = dot(view(λ, j, :), view(λ, k, :))
    end
    return v
end

"""
    _copy_away_from_lowerbd!(sink, source, bd; incr=0.01)

Replace `sink[i]` by `max(source[i], bd[i] + incr)`.  When `bd[i] == -Inf` this simply copies `source[i]`.
"""
function _copy_away_from_lowerbd!(sink, source, bd; incr=0.01)
    for i in eachindex(sink, source, bd)
        @inbounds sink[i] = max(source[i], bd[i] + incr)
    end
    return sink
end

#=  # It appears that this method is not used
"""
    stepsize(tbl::Vector{NamedTuple}, resp::Symbol, pred::Symbol; rev::Bool=false)

Return the stepsize from the last value of `tbl.pred` to increase `resp` by approximately 0.5
"""
function stepsize(tbl::Vector{<:NamedTuple}, resp::Symbol, pred::Symbol)
    ntbl = length(tbl)
    lm1tbl = tbl[ntbl - 1]
    x1 = getproperty(lm1tbl, pred)
    y1 = getproperty(lm1tbl, resp)
    x2 = getproperty(last(tbl), pred)
    y2 = getproperty(last(tbl), resp)
    return (x2 - x1) / (2 * (y2 - y1))
end
=#

function isnondecreasing(spl::SplineInterpolation)
    return all(≥(0), (Derivative(1) * spl).(spl.x))
end
