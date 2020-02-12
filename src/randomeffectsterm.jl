struct RandomEffectsTerm <: AbstractTerm
    lhs::StatsModels.TermOrTerms
    rhs::StatsModels.TermOrTerms
    function RandomEffectsTerm(lhs, rhs)
        if isempty(intersect(StatsModels.termvars(lhs), StatsModels.termvars(rhs)))
            if !isa(
                rhs,
                Union{
                    CategoricalTerm,
                    InteractionTerm{<:NTuple{N,CategoricalTerm} where {N}},
                },
            )
                throw(ArgumentError("blocking variables (those behind |) must be Categorical ($(rhs) is not)"))
            end
            new(lhs, rhs)
        else
            throw(ArgumentError("Same variable appears on both sides of |"))
        end
    end
end

function StatsModels.apply_schema(
    t::FunctionTerm{typeof(/)},
    sch::StatsModels.FullRank,
    Mod::Type{<:MixedModel},
)
    if length(t.args_parsed) ≠ 2
        throw(ArgumentError("malformed nesting term: $t (Exactly two arguments required"))
    end

    first, second = apply_schema.(t.args_parsed, Ref(sch), Mod)
    
    if !(typeof(first) <: CategoricalTerm)
        throw(ArgumentError("nesting terms requires categorical grouping term, got $first.  Manually specify $first as `CategoricalTerm` in hints/contrasts"))
    end

    return first + fulldummy(first) & second
end

RandomEffectsTerm(lhs, rhs::NTuple{2,AbstractTerm}) =
    (RandomEffectsTerm(lhs, rhs[1]), RandomEffectsTerm(lhs, rhs[2]))

Base.show(io::IO, t::RandomEffectsTerm) = print(io, "($(t.lhs) | $(t.rhs))")
StatsModels.is_matrix_term(::Type{RandomEffectsTerm}) = false

function StatsModels.termvars(t::RandomEffectsTerm)
    vcat(StatsModels.termvars(t.lhs), StatsModels.termvars(t.rhs))
end

function StatsModels.apply_schema(
    t::FunctionTerm{typeof(|)},
    schema::StatsModels.FullRank,
    Mod::Type{<:MixedModel},
)
    schema = StatsModels.FullRank(schema.schema)
    lhs, rhs = t.args_parsed
    if !StatsModels.hasintercept(lhs) && !StatsModels.omitsintercept(lhs)
        lhs = InterceptTerm{true}() + lhs
    end
    lhs, rhs = apply_schema.((lhs, rhs), Ref(schema), Mod)
    RandomEffectsTerm(MatrixTerm(lhs), rhs)
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
    for j = 1:S, i = j:S
        push!(inds, m[i, j])
    end
    refs, levels = _ranef_refs(grp, d)

    ReMat{T,S}(
        grp,
        refs,
        levels,
        isa(cnames, String) ? [cnames] : collect(cnames),
        z,
        z,
        LowerTriangular(Matrix{T}(I, S, S)),
        inds,
        adjA(refs, z),
        Matrix{T}(undef, (S, length(levels))),
    )
end


# extract vector of refs from ranef grouping term and data
function _ranef_refs(grp::CategoricalTerm, d::NamedTuple)
    invindex = grp.contrasts.invindex
    refs = convert(Vector{Int32}, getindex.(Ref(invindex), d[grp.sym]))
    refs, grp.contrasts.levels
end

function _ranef_refs(
    grp::InteractionTerm{<:NTuple{N,CategoricalTerm}},
    d::NamedTuple,
) where {N}
    combos = zip(getproperty.(Ref(d), [g.sym for g in grp.terms])...)
    uniques = unique(combos)
    invindex = Dict(x => i for (i, x) in enumerate(uniques))
    refs = convert(Vector{Int32}, getindex.(Ref(invindex), combos))
    refs, uniques
end


# add some syntax to manually promote to full dummy coding
fulldummy(t::AbstractTerm) =
    throw(ArgumentError("can't promote $t (of type $(typeof(t))) to full dummy " *
                        "coding (only CategoricalTerms)"))

function fulldummy(t::CategoricalTerm)
    new_contrasts = StatsModels.ContrastsMatrix(
        StatsModels.FullDummyCoding(),
        t.contrasts.levels,
    )
    t = CategoricalTerm(t.sym, new_contrasts)
end

fulldummy(x) =
    throw(ArgumentError("fulldummy isn't supported outside of a MixedModel formula"))

function StatsModels.apply_schema(
    t::FunctionTerm{typeof(fulldummy)},
    sch::StatsModels.FullRank,
    Mod::Type{<:MixedModel},
)
    fulldummy(apply_schema.(t.args_parsed, Ref(sch), Mod)...)
end


# specify zero correlation
struct ZeroCorr <: AbstractTerm
    term::RandomEffectsTerm
end
StatsModels.is_matrix_term(::Type{ZeroCorr}) = false

"""
    zerocorr(term::RandomEffectsTerm)

Remove correlations between random effects in `term`.
"""
zerocorr(x) = ZeroCorr(x)

function StatsModels.apply_schema(
    t::FunctionTerm{typeof(zerocorr)},
    sch::StatsModels.FullRank,
    Mod::Type{<:MixedModel},
)
    ZeroCorr(apply_schema(t.args_parsed..., sch, Mod))
end

StatsModels.modelcols(t::ZeroCorr, d::NamedTuple) = zerocorr!(modelcols(t.term, d))
