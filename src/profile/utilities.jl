"""
    TableColumns

A structure containing the column names for the numeric part of the profile table.

The struct also contains a Dict giving the column ranges for Symbols like `:σ` and `:β`.
Finally it contains a scratch vector used to accumulate to values in a row of the profile table.
"""
struct TableColumns{T<:AbstractFloat}
    cnames::Tuple
    positions::Dict{Symbol,UnitRange{Int}}
    v::Vector{T}
end

"""
    _generatesyms(tag::Char, len::Integer)

Utility to generate a vector of Symbols of the form :<tag><index> from a tag and a length.

The indices are left-padded with zeros to allow lexicographic sorting.
"""
function _generatesyms(tag::Char, len::Integer)
    return Symbol.(string.(tag, lpad.(Base.OneTo(len), ndigits(len), '0')))
end

function TableColumns(m::LinearMixedModel{T}) where {T}
    nmvec = [:ζ,]
    positions = Dict(:ζ => 1:1)
    lastpos = 1
    sz = m.feterm.rank
    append!(nmvec, _generatesyms('β', sz))
    positions[:β] = (lastpos+1):(lastpos + sz)
    lastpos += sz
    push!(nmvec, :σ)
    lastpos += 1
    positions[:σ] = lastpos:lastpos
    sz = sum(t -> size(t.λ, 1), m.reterms)
    append!(nmvec, _generatesyms('σ', sz))
    positions[:σs] = (lastpos + 1):(lastpos + sz)
    lastpos += sz
    sz = length(m.θ)
    append!(nmvec, _generatesyms('θ', sz))
    positions[:θ] = (lastpos + 1):(lastpos + sz)
    return TableColumns((nmvec...,), positions, Vector{T}(undef, length(nmvec)))
end

function mkrow!(tc::TableColumns{T}, m::LinearMixedModel{T}, ζ::T) where {T}
    @compat (; cnames, positions, v) = tc
    v[1] = ζ
    MixedModels.fixef!(view(v, positions[:β]), m)
    v[first(positions[:σ])] = m.σ
    copyto!(view(v, positions[:σs]), σvals(m))  # re-write this by defining σvals!
    MixedModels.getθ!(view(v, positions[:θ]), m)
    return NamedTuple{cnames, NTuple{length(cnames), T}}((v...,))
end

function parsej(sym::Symbol) # return the index from symbol names like :θ1, :θ01, etc.
    symstr = string(sym)
    return parse(Int, SubString(symstr, nextind(symstr, 1))) # drop first Unicode character and parse as Int
end

"""
    σvals(m::LinearMixedModel)

Return a Tuple of the standard deviation (random effects and per-observation) estimates
"""
function σvals(m::LinearMixedModel{T}) where {T}
    @compat (; σ, reterms) = m
    isone(length(reterms)) && return σvals(only(reterms), σ)
    return (collect(Iterators.flatten(σvals.(reterms, σ)))...,)
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

function collectvals!(v::Vector{T}, m::LinearMixedModel{T}, ζ::T) where {T}
    σ = m.σ
    push!(empty!(v), ζ)
    append!(v, m.β)
    push!(v, σ)
    for t in m.reterms
        append!(v, σvals(t, σ))
    end
    return append!(v, m.θ)
end
