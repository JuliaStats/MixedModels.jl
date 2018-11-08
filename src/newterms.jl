struct RandomEffectsTerm <: AbstractTerm
    lhs::TermOrTerms
    rhs::CategoricalTerm
end

struct FixefTerm <: AbstractTerm
    trms::TermOrTerms
end

israndomeffectsterm(x) = false
israndomeffectsterm(x::RandomEffectsTerm) = true

apply_schema(t::FunctionTerm{typeof(|)}, schema, Mod::Type{<:MixedModel}) = 
    RandomEffectsTerm(apply_schema.(t.args_parsed, Ref(schema), Mod)...)

function apply_schema(terms::NTuple{N,AbstractTerm}, schema, Mod::Type{<:MixedModel}) where N
    fetrms = AbstractTerm[]
    retrms = AbstractTerm[]
    for trm in apply_schema.(terms, Ref(schema), Mod)
        isa(trm, RandomEffectsTerm) ? push!(retrms, trm) : push!(fetrms, trm)
    end
    (fetrms, retrms)
end
