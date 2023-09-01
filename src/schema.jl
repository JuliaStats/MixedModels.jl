using StatsModels:
    FullRank,
    Schema,
    drop_intercept,
    implicit_intercept,
    hasintercept,
    omitsintercept,
    collect_matrix_terms

struct MultiSchema{S}
    base::S
    subs::Dict{Any,S}
end

MultiSchema(s::S) where {S} = MultiSchema(s, Dict{Any,S}())

function StatsModels.apply_schema(t::StatsModels.AbstractTerm, sch::MultiSchema, Ctx::Type)
    return apply_schema(t, sch.base, Ctx)
end

function StatsModels.apply_schema(t::StatsModels.TupleTerm, sch::MultiSchema, Ctx::Type)
    return sum(apply_schema.(t, Ref(sch), Ref(Ctx)))
end

# copied with minimal modifications from StatsModels.jl, in order to wrap the schema
# in MultiSchema.
function StatsModels.apply_schema(t::FormulaTerm, schema::Schema, Mod::Type{<:MixedModel})
    schema = FullRank(schema)

    # Models with the drop_intercept trait do not support intercept terms,
    # usually because they include one implicitly.
    if drop_intercept(Mod)
        if hasintercept(t)
            throw(
                ArgumentError(
                    "Model type $Mod doesn't support intercept " * "specified in formula $t"
                ),
            )
        end
        # start parsing as if we already had the intercept
        push!(schema.already, InterceptTerm{true}())
    elseif implicit_intercept(Mod) && !hasintercept(t) && !omitsintercept(t)
        t = FormulaTerm(t.lhs, InterceptTerm{true}() + t.rhs)
    end

    # only apply rank-promoting logic to RIGHT hand side
    return FormulaTerm(
        apply_schema(t.lhs, schema.schema, Mod),
        collect_matrix_terms(apply_schema(t.rhs, MultiSchema(schema), Mod)),
    )
end

"""
    schematize(f, tbl, contrasts::Dict{Symbol}, Mod=LinearMixedModel)

Find and apply the schema for f in a way that automatically uses `Grouping()`
contrasts when appropriate.

!!! warn
    This is an internal method.
"""
function schematize(f, tbl, contrasts::Dict{Symbol}, Mod=LinearMixedModel)
    # if there is only one term on the RHS, then you don't have an iterator
    # also we want this to be a vector so we can sort later
    rhs = f.rhs isa AbstractTerm ? [f.rhs] : collect(f.rhs)
    fe = filter(!is_randomeffectsterm, rhs)
    # init with lhs so we don't need an extra merge later
    # and so that things work even when we have empty fixed effects
    init = schema(f.lhs, tbl, contrasts)
    sch_fe = mapfoldl(merge, fe; init) do tt
        return schema(tt, tbl, contrasts)
    end
    re = filter(is_randomeffectsterm, rhs)
    sch_re = mapfoldl(merge, re; init) do tt
        # this allows us to control dispatch on a more subtle level
        # and force things to use the schema
        return schema(tt, tbl, contrasts)
    end
    # we want to make sure we don't overwrite any schema
    # determined on the basis of the fixed effects
    # recall: merge prefers the entry in the second argument when there's a duplicate key
    # XXX could we take advantage of MultiSchema here?
    sch = merge(sch_re, sch_fe)

    return apply_schema(f, sch, Mod)
end
