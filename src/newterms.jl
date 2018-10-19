struct RandomEffectsTerm <: AbstractTerm
    lhs::TermOrTerms
    rhs::CategoricalTerm
end

israndomeffectsterm(x) = false
israndomeffectsterm(x::RandomEffectsTerm) = true

apply_schema(t::FunctionTerm{typeof(|)}, schema, Mod::Type{<:MixedModel}) = 
    RandomEffectsTerm(apply_schema.(t.args_parsed, Ref(schema), Mod)...)

function apply_schema(terms::NTuple{N,AbstractTerm}, schema, MixedModel) where N
    terms = apply_schema.(terms, schema, MixedModel)
    (FixefTerm(filter(!israndomeffectsterm, terms)), filter(israndomeffectsterm, terms))
end
