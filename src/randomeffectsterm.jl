struct RandomEffectsTerm <: AbstractTerm
    lhs::TermOrTerms
    rhs::CategoricalTerm
end

Base.show(io::IO, t::RandomEffectsTerm) = print(io, "($(t.lhs) | $(t.rhs))")
StatsModels.is_matrix_term(::Type{RandomEffectsTerm}) = false

function apply_schema(t::FunctionTerm{typeof(|)}, schema, Mod::Type{<:MixedModel})
    lhs, rhs = apply_schema.(t.args_parsed, Ref(schema), Mod)
    RandomEffectsTerm(MatrixTerm(lhs), rhs)
end

StatsModels.termnames(t::RandomEffectsTerm) = string(t.rhs.sym)

struct NoCorrTerm <: AbstractTerm
    reterm::RandomEffectsTerm
end

function nocorr end

function apply_schema(t::FunctionTerm{typeof(nocorr)}, schema, Mod::Type{<:MixedModel})
    args = apply_schema.(t.args_parsed, Ref(schema), Mod)
    isone(length(args)) && isa(args[1], RandomEffectsTerm) || 
        throw(ArgumentError("argument to nocorr must be a random-effect term"))
    NoCorrTerm(args[1])
end

function StatsModels.model_cols(t::RandomEffectsTerm, d::NamedTuple)
    z = Matrix(transpose(model_cols(t.lhs, d)))
    T = eltype(z)
    S = size(z, 1)
    grp = t.rhs
    m = reshape(1:abs2(S), (S, S))
    inds = sizehint!(Int[], (S * (S + 1)) >> 1)
    for j in 1:S, i in j:S
        push!(inds, m[i,j])
    end
    invindex = grp.contrasts.invindex
    refs = getindex.(Ref(invindex), d[grp.sym])
    R = eltype(refs)
    J = R.(1:length(refs))
    II = refs
    if S > 1
        J = repeat(J, inner=S)
        II = R.(vec([(r - 1)*S + j for j in 1:S, r in refs]))
    end
    ReMat(grp, refs, z, z, reinterpret(SVector{S,eltype(z)}, vec(z)),
        LowerTriangular(Matrix{T}(I, S, S)), inds, sparse(II, J, vec(z)))
end
