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

function StatsModels.apply_schema(
    t::StatsModels.AbstractTerm,
    sch::MultiSchema,
    Ctx::Type
)
    apply_schema(t, sch.base, Ctx)
end

function StatsModels.apply_schema(
    t::StatsModels.TupleTerm,
    sch::MultiSchema,
    Ctx::Type
)
    sum(apply_schema.(t, Ref(sch), Ref(Ctx)))
end
    

# copied with minimal modifications from StatsModels.jl, in order to wrap the schema
# in MultiSchema.
function StatsModels.apply_schema(t::FormulaTerm, schema::Schema, Mod::Type{<:MixedModel})
    schema = FullRank(schema)

    # Models with the drop_intercept trait do not support intercept terms,
    # usually because they include one implicitly.
    if drop_intercept(Mod)
        if hasintercept(t)
            throw(ArgumentError("Model type $Mod doesn't support intercept " *
                                "specified in formula $t"))
        end
        # start parsing as if we already had the intercept
        push!(schema.already, InterceptTerm{true}())
    elseif implicit_intercept(Mod) && !hasintercept(t) && !omitsintercept(t)
        t = FormulaTerm(t.lhs, InterceptTerm{true}() + t.rhs)
    end

    # only apply rank-promoting logic to RIGHT hand side
    FormulaTerm(apply_schema(t.lhs, schema.schema, Mod),
                collect_matrix_terms(apply_schema(t.rhs, MultiSchema(schema), Mod)))
end
