"""
    MembershipMatrix{T<:AbstractFloat}

A weights matrix describing the (weighted) membership of observations in the
levels of a multimembership grouping factor.

The `weights` field is stored with **levels as rows and observations as
columns**, i.e. as the transpose of the corresponding section of the
random-effects model matrix `Z`. The `levels` field gives the names of the
levels, corresponding to the rows of `weights`.

See also [`membershipmatrix`](@ref) and [`interactionweights`](@ref) for
convenience constructors.
"""
struct MembershipMatrix{T<:AbstractFloat}
    weights::SparseMatrixCSC{T,Int32}
    levels::Vector{String}
    function MembershipMatrix(
        weights::SparseMatrixCSC{T,Int32}, levels::AbstractVector
    ) where {T}
        levels = string.(vec(levels))
        length(levels) == size(weights, 1) || throw(
            DimensionMismatch(
                "$(length(levels)) levels provided for a weights matrix with $(size(weights, 1)) rows",
            ),
        )
        allunique(levels) ||
            throw(ArgumentError("levels of a membership matrix must be unique"))
        return new{T}(weights, levels)
    end
end

function MembershipMatrix(
    weights::AbstractMatrix; levels=string.(1:size(weights, 1))
)
    T = promote_type(Float64, float(eltype(weights)))
    return MembershipMatrix(convert(SparseMatrixCSC{T,Int32}, sparse(weights)), levels)
end

Base.size(mm::MembershipMatrix) = size(mm.weights)
Base.size(mm::MembershipMatrix, i::Integer) = size(mm.weights, i)

nlevs(mm::MembershipMatrix) = length(mm.levels)

DataAPI.levels(mm::MembershipMatrix) = mm.levels

function Base.show(io::IO, ::MIME"text/plain", mm::MembershipMatrix{T}) where {T}
    q, n = size(mm)
    println(io, "MembershipMatrix{$T} with $q levels and $n observations")
    memb = _membership_counts(mm)
    print(
        io,
        "memberships per observation: min $(minimum(memb)), mean $(round(sum(memb) / n; digits=2)), max $(maximum(memb))",
    )
    return nothing
end

function _membership_counts(mm::MembershipMatrix)
    W = mm.weights
    return diff(W.colptr)
end

function _normalize_cols!(W::SparseMatrixCSC{T}) where {T}
    nz = nonzeros(W)
    @inbounds for j in axes(W, 2)
        rng = nzrange(W, j)
        s = sum(k -> nz[k], rng; init=zero(T))
        if !iszero(s)
            for k in rng
                nz[k] /= s
            end
        end
    end
    return W
end

"""
    membershipmatrix(memberships::AbstractVector{<:AbstractString};
                     delim=",", normalize=false)
    membershipmatrix(cols::AbstractVector...; normalize=false)

Construct a [`MembershipMatrix`](@ref) from a description of the memberships
of each observation.

In the first form, each element of `memberships` is a `delim`-separated list
of the levels that the corresponding observation belongs to (whitespace
around the delimiter is stripped). A level mentioned more than once for the
same observation accumulates weight.

In the second form, each argument is a vector giving (at most) one membership
per observation; `missing` entries are skipped.

The levels are sorted lexicographically. If `normalize=true`, the weights in
each observation's column are scaled to sum to one.
"""
function membershipmatrix(
    memberships::AbstractVector{<:AbstractString}; delim=",", normalize::Bool=false
)
    n = length(memberships)
    I = Int32[]
    J = Int32[]
    membs = [string.(strip.(split(el, delim))) for el in memberships]
    levs = sort!(unique!(reduce(vcat, membs; init=String[])))
    invindex = Dict(l => Int32(i) for (i, l) in enumerate(levs))
    for (j, memb) in enumerate(membs)
        for l in memb
            push!(I, invindex[l])
            push!(J, Int32(j))
        end
    end
    W = sparse(I, J, ones(length(I)), length(levs), n)
    normalize && _normalize_cols!(W)
    return MembershipMatrix(W; levels=levs)
end

function membershipmatrix(cols::AbstractVector...; normalize::Bool=false)
    n = length(first(cols))
    all(==(n) ∘ length, cols) ||
        throw(DimensionMismatch("all membership columns must have the same length"))
    I = Int32[]
    J = Int32[]
    levs = sort!(unique!(string.(collect(skipmissing(reduce(vcat, cols))))))
    invindex = Dict(l => Int32(i) for (i, l) in enumerate(levs))
    for col in cols
        for (j, el) in enumerate(col)
            ismissing(el) && continue
            push!(I, invindex[string(el)])
            push!(J, Int32(j))
        end
    end
    W = sparse(I, J, ones(length(I)), length(levs), n)
    normalize && _normalize_cols!(W)
    return MembershipMatrix(W; levels=levs)
end

"""
    interactionweights(a::MembershipMatrix, b::MembershipMatrix)

Construct the [`MembershipMatrix`](@ref) corresponding to the interaction of
two multimembership grouping factors, i.e. the column-wise Khatri-Rao product
of the two weights matrices. The levels of the result are named `"la & lb"`
for each pair of levels `la` of `a` and `lb` of `b`.

This is useful for constructing nested or interaction groupings involving a
multimembership factor, which cannot be expressed directly in the formula.
A single-membership factor can be converted to a `MembershipMatrix` with
[`membershipmatrix`](@ref).
"""
function interactionweights(a::MembershipMatrix{T}, b::MembershipMatrix{T}) where {T}
    Wa = a.weights
    Wb = b.weights
    n = size(Wa, 2)
    size(Wb, 2) == n || throw(
        DimensionMismatch("membership matrices must have the same number of observations")
    )
    nb = nlevs(b)
    I = Int32[]
    J = Int32[]
    V = T[]
    arv, anz = rowvals(Wa), nonzeros(Wa)
    brv, bnz = rowvals(Wb), nonzeros(Wb)
    for j in axes(Wa, 2)
        for ka in nzrange(Wa, j)
            for kb in nzrange(Wb, j)
                push!(I, (arv[ka] - 1) * nb + brv[kb])
                push!(J, Int32(j))
                push!(V, anz[ka] * bnz[kb])
            end
        end
    end
    levs = [string(la, " & ", lb) for la in a.levels for lb in b.levels]
    W = sparse(I, J, V, nlevs(a) * nb, n)
    return MembershipMatrix(W; levels=levs)
end

"""
    MultimembershipReMat(rt::ReMat, mm::MembershipMatrix)

Construct a [`MultimembershipReMat`](@ref) by replacing the level-membership
structure of a (placeholder) `ReMat` with the weighted memberships in `mm`.

The left-hand side of the random-effects term (`z`, `cnames`, `λ`) is reused;
the sparse model matrix is the Khatri-Rao product of `mm.weights` and `z`.
"""
function MultimembershipReMat(rt::ReMat{T,S}, mm::MembershipMatrix) where {T,S}
    W = convert(SparseMatrixCSC{T,Int32}, mm.weights)
    n = size(rt.z, 2)
    size(W, 2) == n || throw(
        DimensionMismatch(
            "membership matrix for $(fname(rt)) has $(size(W, 2)) observations but the data have $n rows",
        ),
    )
    adjA = _khatrirao(W, rt.z)
    return MultimembershipReMat{T,S}(
        rt.trm,
        mm.levels,
        rt.cnames,
        rt.z,
        rt.z,
        rt.λ,
        rt.inds,
        adjA,
        adjA,
        Matrix{T}(undef, S, nlevs(mm)),
    )
end

function _khatrirao(W::SparseMatrixCSC{T,Int32}, z::Matrix{T}) where {T}
    S = size(z, 1)
    wI, wJ, wV = findnz(W)
    I = Vector{Int32}(undef, length(wV) * S)
    J = similar(I)
    V = Vector{T}(undef, length(wV) * S)
    idx = 0
    for k in eachindex(wV)
        j = wJ[k]
        for s in 1:S
            idx += 1
            I[idx] = (wI[k] - 1) * S + s
            J[idx] = j
            V[idx] = wV[k] * z[s, j]
        end
    end
    return sparse(I, J, V, S * size(W, 1), size(W, 2))
end

_normalize_memberships(::Nothing) = nothing

function _normalize_memberships(memberships)
    return Dict{Symbol,MembershipMatrix}(
        Symbol(k) => (v isa MembershipMatrix ? v : MembershipMatrix(v)) for
        (k, v) in pairs(memberships)
    )
end

# inject a constant placeholder column for each membership grouping variable
# that is not already a column of the table; the placeholder's levels/refs are
# discarded when the corresponding ReMat is replaced by a MultimembershipReMat
function _inject_membership_columns(
    tbl::Tables.ColumnTable, memberships::Dict{Symbol,MembershipMatrix}
)
    nrows = length(first(tbl))
    for (g, mm) in memberships
        if g in Tables.columnnames(tbl)
            throw(
                ArgumentError(
                    "membership grouping variable $g is also a column in the data; " *
                    "please use a distinct name for the multimembership grouping factor",
                ),
            )
        end
        tbl = merge(tbl, NamedTuple{(g,)}((fill(first(mm.levels), nrows),)))
    end
    return tbl
end

# replace placeholder ReMats by MultimembershipReMats for every grouping
# variable with an associated MembershipMatrix
function _substitute_memberships!(
    reterms::Vector{<:AbstractReMat}, memberships::Dict{Symbol,MembershipMatrix}
)
    matched = Set{Symbol}()
    for (i, rt) in enumerate(reterms)
        g = fname(rt)
        if haskey(memberships, g)
            rt isa ReMat || throw(
                ArgumentError("cannot attach a membership matrix to $(typeof(rt))")
            )
            reterms[i] = MultimembershipReMat(rt, memberships[g])
            push!(matched, g)
        end
    end
    unmatched = setdiff(keys(memberships), matched)
    isempty(unmatched) || throw(
        ArgumentError(
            "membership matrices provided for $(join(unmatched, ", ")) but no corresponding random-effects grouping variable was found in the formula",
        ),
    )
    return reterms
end

# simulate!/predict with newdata reconstruct the random-effects design from
# scratch, so models with multimembership terms need a new membership matrix
function _check_memberships_newdata(m::MixedModel, memberships)
    if memberships === nothing && any(ismultimember, m.reterms)
        throw(
            ArgumentError(
                "simulating or predicting from new data for a model with " *
                "multimembership terms requires the `memberships` keyword argument",
            ),
        )
    end
    return nothing
end

function _subset_memberships(
    memberships::Dict{Symbol,MembershipMatrix}, nonmissings::AbstractVector{Bool}, n::Int
)
    for (g, mm) in memberships
        size(mm, 2) == n || throw(
            DimensionMismatch(
                "membership matrix for $g has $(size(mm, 2)) observations but the data have $n rows",
            ),
        )
    end
    all(nonmissings) && return memberships
    return Dict{Symbol,MembershipMatrix}(
        g => MembershipMatrix(mm.weights[:, nonmissings], mm.levels) for
        (g, mm) in memberships
    )
end
