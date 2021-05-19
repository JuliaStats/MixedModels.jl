
"""
    struct Grouping <: StatsModels.AbstractContrasts end

A placeholder type to indicate that a categorical variable is only used for
grouping and not for contrasts.  When creating a [`CategoricalTerm`](@ref), this
skips constructing the contrasts matrix which makes it robust to large numbers
of levels, while still holding onto the vector of levels and constructing the
level-to-index mapping (`invindex` field of the [`ContrastsMatrix`](@ref).).

Note that calling `modelcols` on a `CategoricalTerm{Grouping}` is an error.

# Examples

```julia
julia> schema((; grp = string.(1:100_000)))
# out-of-memory error

julia> schema((; grp = string.(1:100_000)), Dict(:grp => Grouping()))
```
"""
struct Grouping <: StatsModels.AbstractContrasts
end

# return an empty matrix
StatsModels.contrasts_matrix(::Grouping, baseind, n) = zeros(0,0)
StatsModels.termnames(::Grouping, levels::AbstractVector, baseind::Integer) = levels

# this is needed until StatsModels stops assuming all contrasts have a .levels field
Base.getproperty(g::Grouping, prop::Symbol) =
    prop == :levels ? nothing : getfield(g, prop)

# special-case categorical terms with Grouping contrasts.
StatsModels.modelcols(::CategoricalTerm{Grouping}, d::NamedTuple) =
    error("can't create model columns directly from a Grouping term")
