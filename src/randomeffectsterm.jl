struct RandomEffectsTerm <: AbstractTerm
    lhs::StatsModels.TermOrTerms
    rhs::StatsModels.TermOrTerms
end

Base.show(io::IO, t::RandomEffectsTerm) = print(io, "($(t.lhs) | $(t.rhs))")
StatsModels.is_matrix_term(::Type{RandomEffectsTerm}) = false

function StatsModels.apply_schema(t::FunctionTerm{typeof(|)}, schema::StatsModels.FullRank,
        Mod::Type{<:MixedModel})
    lhs, rhs = apply_schema.(t.args_parsed, Ref(schema), Mod)
    RandomEffectsTerm(MatrixTerm(lhs), rhs)
end

struct NoCorrTerm <: AbstractTerm
    reterm::RandomEffectsTerm
end

function nocorr end

function StatsModels.apply_schema(t::FunctionTerm{typeof(nocorr)}, schema,
        Mod::Type{<:MixedModel})
    args = apply_schema.(t.args_parsed, Ref(schema), Mod)
    isone(length(args)) && isa(args[1], RandomEffectsTerm) || 
        throw(ArgumentError("argument to nocorr must be a random-effect term"))
    NoCorrTerm(args[1])
end

function StatsModels.modelcols(t::RandomEffectsTerm, d::NamedTuple)
    lhs = t.lhs
    z = Matrix(transpose(modelcols(lhs, d)))
    cnames = coefnames(lhs)
    T = eltype(z)
    S = size(z, 1)
    grp = t.rhs
    m = reshape(1:abs2(S), (S, S))
    inds = sizehint!(Int[], (S * (S + 1)) >> 1)
    for j in 1:S, i in j:S
        push!(inds, m[i,j])
    end
    invindex = grp.contrasts.invindex
    refs = convert(Vector{Int32}, getindex.(Ref(invindex), d[grp.sym]))
    J = Int32.(1:length(refs))
    II = refs
    if S > 1
        J = repeat(J, inner=S)
        II = Int32.(vec([(r - 1)*S + j for j in 1:S, r in refs]))
    end
    ReMat{T,S}(grp, refs, isa(cnames, String) ? [cnames] : collect(cnames), 
        z, z, LowerTriangular(Matrix{T}(I, S, S)), inds,
        sparse(II, J, vec(z)), Matrix{T}(undef, (S, length(invindex))))
end
