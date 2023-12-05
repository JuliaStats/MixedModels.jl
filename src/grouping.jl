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

# this is needed until StatsModels stops assuming all contrasts have a .levels field
Base.getproperty(g::Grouping, prop::Symbol) = prop == :levels ? nothing : getfield(g, prop)

# special-case categorical terms with Grouping contrasts.
function StatsModels.modelcols(::CategoricalTerm{Grouping}, d::NamedTuple)
    return error("can't create model columns directly from a Grouping term")
end

function StatsModels.ContrastsMatrix(
    contrasts::Grouping, levels::AbstractVector
)
    return StatsModels.ContrastsMatrix(zeros(0, 0), levels, levels, contrasts)
end

# this arises when there's an interaction as a grouping variable without a corresponding
# non-interaction grouping, e.g. urban&dist in the contra dataset
# adapted from https://github.com/JuliaStats/StatsModels.jl/blob/463eb0acb49bc5428374d749c4da90ea2a6c74c4/src/schema.jl#L355-L372
function StatsModels.apply_schema(
    t::CategoricalTerm{Grouping},
    schema::FullRank,
    ::Type{<:MixedModel},
    context::AbstractTerm,
)
    aliased = drop_term(context, t)
    #@debug "$t in context of $context: aliases $aliased\n  seen already: $(schema.already)"
    for seen in schema.already
        if StatsModels.symequal(aliased, seen)
            #@debug "  aliased term already present: $seen"
            return t
        end
    end
    # aliased term not seen already:
    # add aliased term to already seen:
    push!(schema.already, aliased)
    # repair:
    new_contrasts = StatsModels.ContrastsMatrix(Grouping(), t.contrasts.levels)
    t = CategoricalTerm(t.sym, new_contrasts)
    #@debug "  aliased term absent, repairing: $t"
    return t
end
