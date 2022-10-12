
"""
    struct Grouping <: StatsModels.AbstractContrasts end

A placeholder type to indicate that a categorical variable is only used for
grouping and not for contrasts.  When creating a `CategoricalTerm`, this
skips constructing the contrasts matrix which makes it robust to large numbers
of levels, while still holding onto the vector of levels and constructing the
level-to-index mapping (`invindex` field of the `ContrastsMatrix`.).

Note that calling `modelcols` on a `CategoricalTerm{Grouping}` is an error.

# Examples

```julia
julia> schema((; grp = string.(1:100_000)))
# out-of-memory error

julia> schema((; grp = string.(1:100_000)), Dict(:grp => Grouping()))
```
"""
struct Grouping <: StatsModels.AbstractContrasts end

# return an empty matrix
StatsModels.contrasts_matrix(::Grouping, baseind, n) = zeros(0, 0)
StatsModels.termnames(::Grouping, levels::AbstractVector, baseind::Integer) = levels

# this is needed until StatsModels stops assuming all contrasts have a .levels field
Base.getproperty(g::Grouping, prop::Symbol) = prop == :levels ? nothing : getfield(g, prop)

# special-case categorical terms with Grouping contrasts.
function StatsModels.modelcols(::CategoricalTerm{Grouping}, d::NamedTuple)
    return error("can't create model columns directly from a Grouping term")
end

# copied from StatsModels@463eb0a
function StatsModels.ContrastsMatrix(
    contrasts::Grouping, levels::AbstractVector{T}
) where {T}

    # if levels are defined on contrasts, use those, validating that they line up.
    # what does that mean? either:
    #
    # 1. DataAPI.levels(contrasts) == levels (best case)
    # 2. data levels missing from contrast: would generate empty/undefined rows.
    #    better to filter data frame first
    # 3. contrast levels missing from data: would have empty columns, generate a
    #    rank-deficient model matrix.
    c_levels = something(DataAPI.levels(contrasts), levels)

    mismatched_levels = symdiff(c_levels, levels)
    if !isempty(mismatched_levels)
        throw(
            ArgumentError(
                "contrasts levels not found in data or vice-versa: " *
                "$mismatched_levels." *
                "\n  Data levels ($(eltype(levels))): $levels." *
                "\n  Contrast levels ($(eltype(c_levels))): $c_levels",
            ),
        )
    end

    # do conversion AFTER checking for levels so users get a nice error message
    # when they've made a mistake with the level types
    c_levels = convert(Vector{T}, c_levels)

    n = length(c_levels)
    # not validating this allows for prediction of only a single level of the grouping factor
    # if n == 0
    #     throw(ArgumentError("empty set of levels found (need at least two to compute " *
    #                         "contrasts)."))
    # elseif n == 1
    #     throw(ArgumentError("only one level found: $(c_levels[1]) (need at least two to " *
    #                         "compute contrasts)."))
    # end
    # find index of base level. use baselevel(contrasts), then default (1).
    # base_level = baselevel(contrasts)
    # baseind = base_level === nothing ?
    #           1 :
    #           findfirst(isequal(base_level), c_levels)
    # if baseind === nothing
    #     throw(ArgumentError("base level $(base_level) not found in levels " *
    #                         "$c_levels."))
    # end

    base_level = nothing
    baseind = 1

    tnames = StatsModels.termnames(contrasts, c_levels, baseind)
    mat = StatsModels.contrasts_matrix(contrasts, baseind, n)
    return StatsModels.ContrastsMatrix(mat, tnames, c_levels, contrasts)
end
