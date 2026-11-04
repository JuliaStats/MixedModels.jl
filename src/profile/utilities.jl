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
    getθ!(view(v, positions[:θ]), m)
    length(corrpos) > 0 && ρvals!(view(v, positions[:ρs]), corrpos, m)
    return NamedTuple{cnames,NTuple{N,T}}((v...,))
end

"""
    _ζ(objective::T, fmin::T, neg::Bool, sym::Symbol, value) where {T}

Return the profile ζ, `sqrt(objective - fmin)`, negated if `neg` is `true` (i.e. when the
profiled parameter is below its estimate).

A negative difference within a small tolerance is treated as zero. A larger negative
difference means that `fmin` is not the minimum of the objective and an `ArgumentError`
naming the parameter `sym` and its `value` is thrown.
"""
function _ζ(objective::T, fmin::T, neg::Bool, sym::Symbol, value) where {T}
    δ = objective - fmin
    if δ < 0
        δ ≥ -sqrt(eps(T)) * max(one(T), abs(fmin)) || throw(
            ArgumentError(
                "objective at $sym = $value is $(-δ) below the minimum $fmin; " *
                "the model fit may not have converged. Try refitting with tighter tolerances.",
            ),
        )
        δ = zero(T)
    end
    ζ = sqrt(δ)
    return neg ? -ζ : ζ
end

"""
    _profileobjective!(m::LinearMixedModel, θ)

Return `objective!(m, θ)`, or `m.optsum.finitial` if the factorization fails because it is
not positive definite.

This mirrors the objective used for fitting the model. The optimizers in the profiles can
move into regions of the parameter space where there is not enough shrinkage for the
factorization, and `finitial` is generally a value that the optimizer won't view as an
optimum.

!!! note
    This method is internal.
"""
function _profileobjective!(m::LinearMixedModel, θ)
    return try
        objective!(m, θ)
    catch ex
        ex isa PosDefException || rethrow()
        m.optsum.finitial
    end
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
    for (ii, (i, j, k)) in enumerate(corrpos)
        λ = reterms[i].λ
        rowj = view(λ, j, :)
        rowk = view(λ, k, :)
        nrm = norm(rowj) * norm(rowk)
        # a row of zeros has no defined correlation; use zero, as in `rownormalize`
        v[ii] = iszero(nrm) ? zero(T) : dot(rowj, rowk) / nrm
    end
    return v
end

function isnondecreasing(spl::SplineInterpolation)
    return all(≥(0), (Derivative(1) * spl).(spl.x))
end
