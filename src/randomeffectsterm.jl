abstract type AbstractReTerm <: AbstractTerm end

struct RandomEffectsTerm <: AbstractReTerm
    lhs::StatsModels.TermOrTerms
    rhs::StatsModels.TermOrTerms
end

# TODO: consider overwriting | with our own function that can be
# imported with (a la FilePathsBase.:/)
# using MixedModels: |
# to avoid conflicts with definitions in other packages...
Base.:|(a::StatsModels.TermOrTerms, b::StatsModels.TermOrTerms) = RandomEffectsTerm(a, b)

# expand (lhs | a + b) to (lhs | a) + (lhs | b)
RandomEffectsTerm(lhs::StatsModels.TermOrTerms, rhs::NTuple{2,AbstractTerm}) =
    (RandomEffectsTerm(lhs, rhs[1]), RandomEffectsTerm(lhs, rhs[2]))

Base.show(io::IO, t::RandomEffectsTerm) = print(io, "($(t.lhs) | $(t.rhs))")
StatsModels.is_matrix_term(::Type{RandomEffectsTerm}) = false

function StatsModels.termvars(t::RandomEffectsTerm)
    vcat(StatsModels.termvars(t.lhs), StatsModels.termvars(t.rhs))
end

# | in MixedModel formula -> RandomEffectsTerm
function StatsModels.apply_schema(
    t::FunctionTerm{typeof(|)},
    schema::MultiSchema{StatsModels.FullRank},
    Mod::Type{<:MixedModel}
)
    lhs, rhs = t.args_parsed

    isempty(intersect(StatsModels.termvars(lhs), StatsModels.termvars(rhs))) ||
        throw(ArgumentError("Same variable appears on both sides of |"))

    apply_schema(RandomEffectsTerm(lhs, rhs), schema, Mod)
end

# make a potentially untyped RandomEffectsTerm concrete
function StatsModels.apply_schema(
    t::RandomEffectsTerm,
    schema::MultiSchema{StatsModels.FullRank},
    Mod::Type{<:MixedModel}
)
    lhs, rhs = t.lhs, t.rhs

    # get a schema that's specific for the grouping (RHS), creating one if needed
    schema = get!(schema.subs, rhs, StatsModels.FullRank(schema.base.schema))

    # handle intercept in LHS (including checking schema for intercept in another term)
    if (
        !StatsModels.hasintercept(lhs) &&
        !StatsModels.omitsintercept(lhs) &&
        ConstantTerm(1) ∉ schema.already &&
        InterceptTerm{true}() ∉ schema.already
    )
        lhs = InterceptTerm{true}() + lhs
    end

    lhs, rhs = apply_schema.((lhs, rhs), Ref(schema), Mod)

    # check whether grouping terms are categorical or interaction of categorical
    rhs isa CategoricalTerm ||
        rhs isa InteractionTerm{<:NTuple{N,CategoricalTerm} where {N}} ||
        throw(ArgumentError("blocking variables (those behind |) must be Categorical ($(rhs) is not)"))
    
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




# add some syntax to manually promote to full dummy coding
fulldummy(t::AbstractTerm) =
    throw(ArgumentError("can't promote $t (of type $(typeof(t))) to full dummy " *
                        "coding (only CategoricalTerms)"))

"""
    fulldummy(term::CategoricalTerm)

Assign "contrasts" that include all indicator columns (dummy variables) and an intercept column.

This will result in an under-determined set of contrasts, which is not a problem in the random
effects because of the regularization, or "shrinkage", of the conditional modes.

The interaction of `fulldummy` with complex random effects is subtle and complex with numerous 
potential edge cases. As we discover these edge cases, we will document and determine their 
behavior. Until such time, please check the model summary to verify that the expansion is 
working as you expected. If it is not, please report a use case on GitHub.
"""
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
struct ZeroCorr <: AbstractReTerm
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
    sch::MultiSchema,
    Mod::Type{<:MixedModel},
)
    ZeroCorr(apply_schema(t.args_parsed..., sch, Mod))
end

StatsModels.modelcols(t::ZeroCorr, d::NamedTuple) = zerocorr!(modelcols(t.term, d))
