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
function RandomEffectsTerm(lhs::StatsModels.TermOrTerms, rhs::NTuple{2,AbstractTerm})
    return (RandomEffectsTerm(lhs, rhs[1]), RandomEffectsTerm(lhs, rhs[2]))
end

Base.show(io::IO, t::RandomEffectsTerm) = Base.show(io, MIME"text/plain"(), t)

function Base.show(io::IO, ::MIME"text/plain", t::RandomEffectsTerm)
    return print(io, "($(t.lhs) | $(t.rhs))")
end
StatsModels.is_matrix_term(::Type{RandomEffectsTerm}) = false

function StatsModels.termvars(t::RandomEffectsTerm)
    return vcat(StatsModels.termvars(t.lhs), StatsModels.termvars(t.rhs))
end

function StatsModels.terms(t::RandomEffectsTerm)
    return union(StatsModels.terms(t.lhs), StatsModels.terms(t.rhs))
end

schema(t, data, hints) = StatsModels.schema(t, data, hints)

function schema(t::FunctionTerm{typeof(|)}, data, hints::Dict{Symbol})
    sch = schema(t.args[1], data, hints)
    vars = StatsModels.termvars.(t.args[2])
    # in the event that someone has x|x, then the Grouping()
    # gets overwrriten by the broader schema BUT
    # that doesn't matter because we detect and throw an error
    # for that in apply_schema
    grp_hints = Dict(rr => Grouping() for rr in vars)
    return merge(schema(t.args[2], data, grp_hints), sch)
end

is_randomeffectsterm(::Any) = false
is_randomeffectsterm(::AbstractReTerm) = true
# RE with free covariance structure
is_randomeffectsterm(::FunctionTerm{typeof(|)}) = true
# not zerocorr() or the like
is_randomeffectsterm(tt::FunctionTerm) = is_randomeffectsterm(tt.args[1])

# | in MixedModel formula -> RandomEffectsTerm
function StatsModels.apply_schema(
    t::FunctionTerm{typeof(|)},
    schema::MultiSchema{StatsModels.FullRank},
    Mod::Type{<:MixedModel},
)
    lhs, rhs = t.args

    isempty(intersect(StatsModels.termvars(lhs), StatsModels.termvars(rhs))) ||
        throw(ArgumentError("Same variable appears on both sides of |"))

    return apply_schema(RandomEffectsTerm(lhs, rhs), schema, Mod)
end

# allowed types (or tuple thereof) for blocking variables (RHS of |):
const GROUPING_TYPE = Union{
    <:CategoricalTerm,<:InteractionTerm{<:NTuple{N,CategoricalTerm} where {N}}
}
check_re_group_type(term::GROUPING_TYPE) = true
check_re_group_type(term::Tuple) = all(check_re_group_type, term)
check_re_group_type(x) = false

_unprotect(x) = x
for op in StatsModels.SPECIALS
    @eval _unprotect(t::FunctionTerm{typeof($op)}) = t.f(_unprotect.(t.args)...)
end

# make a potentially untyped RandomEffectsTerm concrete
function StatsModels.apply_schema(
    t::RandomEffectsTerm, schema::MultiSchema{StatsModels.FullRank}, Mod::Type{<:MixedModel}
)
    # we need to do this here because the implicit intercept dance has to happen
    # _before_ we apply_schema, which is where :+ et al. are normally
    # unprotected.  I tried to finagle a way around this (using yet another
    # schema wrapper type) but it ends up creating way too many potential/actual
    # method ambiguities to be a good idea.
    lhs, rhs = _unprotect(t.lhs), t.rhs

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
    check_re_group_type(rhs) || throw(
        ArgumentError(
            "blocking variables (those behind |) must be Categorical ($(rhs) is not)"
        ),
    )

    return RandomEffectsTerm(MatrixTerm(lhs), rhs)
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
        push!(inds, m[i, j])
    end
    refs, levels = _ranef_refs(grp, d)

    return ReMat{T,S}(
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
    return refs, grp.contrasts.levels
end

function _ranef_refs(
    grp::InteractionTerm{<:NTuple{N,CategoricalTerm}}, d::NamedTuple
) where {N}
    combos = zip(getproperty.(Ref(d), [g.sym for g in grp.terms])...)
    uniques = unique(combos)
    invindex = Dict(x => i for (i, x) in enumerate(uniques))
    refs = convert(Vector{Int32}, getindex.(Ref(invindex), combos))
    return refs, uniques
end

# TODO: remove all of this and either
# - require users to use RegressionFormulae.jl
# - add a dependency on RegressionFormulae.jl

function StatsModels.apply_schema(
    t::FunctionTerm{typeof(/)}, sch::StatsModels.FullRank, Mod::Type{<:MixedModel}
)
    if length(t.args) ≠ 2
        throw(ArgumentError("malformed nesting term: $t (Exactly two arguments required"))
    end

    first, second = apply_schema.(t.args, Ref(sch), Mod)

    if !(typeof(first) <: CategoricalTerm)
        throw(
            ArgumentError(
                "nesting terms requires categorical grouping term, got $first.  Manually specify $first as `CategoricalTerm` in hints/contrasts"
            ),
        )
    end

    return first + fulldummy(first) & second
end

# add some syntax to manually promote to full dummy coding
function fulldummy(t::AbstractTerm)
    return throw(
        ArgumentError(
            "can't promote $t (of type $(typeof(t))) to full dummy " *
            "coding (only CategoricalTerms)",
        ),
    )
end

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
        StatsModels.FullDummyCoding(), t.contrasts.levels
    )
    return t = CategoricalTerm(t.sym, new_contrasts)
end

function fulldummy(x)
    return throw(ArgumentError("fulldummy isn't supported outside of a MixedModel formula"))
end

function StatsModels.apply_schema(
    t::FunctionTerm{typeof(fulldummy)}, sch::StatsModels.FullRank, Mod::Type{<:MixedModel}
)
    return fulldummy(apply_schema.(t.args, Ref(sch), Mod)...)
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

# for schema extraction (from runtime-created zerocorr)
StatsModels.terms(t::ZeroCorr) = StatsModels.terms(t.term)
StatsModels.termvars(t::ZeroCorr) = StatsModels.termvars(t.term)
StatsModels.degree(t::ZeroCorr) = StatsModels.degree(t.term)
# dirty rotten no good ugly hack: make sure zerocorr ranef terms sort appropriately
# cf https://github.com/JuliaStats/StatsModels.jl/blob/41b025409af03c0e019591ac6e817b22efbb4e17/src/terms.jl#L421-L422
StatsModels.degree(t::FunctionTerm{typeof(zerocorr)}) = StatsModels.degree(only(t.args))

Base.show(io::IO, t::ZeroCorr) = Base.show(io, MIME"text/plain"(), t)
function Base.show(io::IO, ::MIME"text/plain", t::ZeroCorr)
    # ranefterms already show with parens
    return print(io, "zerocorr", t.term)
end

function schema(t::FunctionTerm{typeof(zerocorr)}, data, hints::Dict{Symbol})
    return schema(only(t.args), data, hints)
end

function StatsModels.apply_schema(
    t::FunctionTerm{typeof(zerocorr)}, sch::MultiSchema, Mod::Type{<:MixedModel}
)
    return ZeroCorr(apply_schema(only(t.args), sch, Mod))
end

function StatsModels.apply_schema(t::ZeroCorr, sch::MultiSchema, Mod::Type{<:MixedModel})
    return ZeroCorr(apply_schema(t.term, sch, Mod))
end

StatsModels.modelcols(t::ZeroCorr, d::NamedTuple) = zerocorr!(modelcols(t.term, d))

function Base.getproperty(x::ZeroCorr, s::Symbol)
    return s == :term ? getfield(x, s) : getproperty(x.term, s)
end
