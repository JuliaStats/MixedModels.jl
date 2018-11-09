struct RandomEffectsTerm <: AbstractTerm
    lhs::TermOrTerms
    rhs::CategoricalTerm
end

struct ReMat{T,R,S} <: AbstractMatrix{T}
    trm::CategoricalTerm
    refs::Vector{R}
    z::Matrix{T}
    wtz::Matrix{T}
    wtzv::Base.ReinterpretArray{SVector{S,T}}
end

function apply_schema(t::FunctionTerm{typeof(|)}, schema, Mod::Type{<:MixedModel})
    lhs, rhs = apply_schema.(t.args_parsed, Ref(schema), Mod)
    RandomEffectsTerm(isa(lhs, Tuple) ? apply_schema.(lhs, Ref(schema), Mod) : lhs, rhs)
end

StatsModels.termnames(t::RandomEffectsTerm) = string(t.rhs.sym)

function StatsModels.model_cols(t::RandomEffectsTerm, d::NamedTuple)
    z = Matrix(transpose(model_cols(t.lhs, d)))
    k = size(z, 1)
    grp = t.rhs
    ReMat(grp, getindex.(Ref(t.rhs.contrasts.invindex), d[grp.sym]), z, z, reinterpret(SVector{k,eltype(z)}, vec(z)))
end
