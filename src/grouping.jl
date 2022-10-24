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

function _grouping_vars(f::FormulaTerm)
    # if there is only one term on the RHS, then you don't have an iterator
    rhs = f.rhs isa AbstractTerm ? (f.rhs,) : f.rhs
    re = filter(x -> x isa RE_FUNCTION_TERM, rhs)
    grping = unique!(mapreduce(x -> x.args_parsed[end], vcat, re; init=[]))
    # XXX how to handle interaction terms in Grouping?
    # for now, we just don't.
    return grping = mapreduce(vcat, grping; init=Symbol[]) do g
        hasproperty(g, :sym) && return [g.sym]
        return collect(getproperty.(terms(grping[1]), :sym))
    end
end
