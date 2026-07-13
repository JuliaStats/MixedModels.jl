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

Base.show(io::IO, t::RandomEffectsTerm) = Base.show(io, MIME("text/plain"), t)

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

function schema(t::AbstractReTerm, data, hints::Dict{Symbol})
    sch = schema(t.lhs, data, hints)
    vars = StatsModels.termvars.(t.rhs)
    # in the event that someone has x|x, then the Grouping()
    # gets overwritten by the broader schema BUT
    # that doesn't matter because we detect and throw an error
    # for that in apply_schema
    grp_hints = Dict(rr => Grouping() for rr in vars)
    return merge(schema(t.rhs, data, grp_hints), sch)
end

function schema(t::FunctionTerm{typeof(|)}, data, hints::Dict{Symbol})
    re = RandomEffectsTerm(t.args[1], t.args[2])
    return schema(re, data, hints)
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
    lhs = apply_schema(lhs, schema, Mod)
    rhs = apply_schema(rhs, schema, Mod)

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

# specify a constrained covariance structure
"""
    StructuredReTerm{C}

A random-effects term with a constrained covariance structure.

The type parameter `C` tags the covariance structure imposed on the wrapped
`RandomEffectsTerm` via [`structure!`](@ref) when the model matrices are
constructed. All term-level behavior is independent of `C`; everything specific
to the structure lives in the [`CovarianceStructure`](@ref) installed on the
resulting `ReMat`.
"""
struct StructuredReTerm{C} <: AbstractReTerm
    term::RandomEffectsTerm
end

StatsModels.is_matrix_term(::Type{<:StructuredReTerm}) = false

"""
    ZeroCorr

Alias for the `StructuredReTerm` created by [`zerocorr`](@ref).
"""
const ZeroCorr = StructuredReTerm{ZeroCorrStruct}

"""
    zerocorr(term::RandomEffectsTerm)

Remove correlations between random effects in `term`.
"""
zerocorr(x) = ZeroCorr(x)

"""
    homdiag(term::RandomEffectsTerm)

Constrain the covariance matrix of the random effects in `term` to a multiple
of the identity, i.e. independent random effects with a common variance.

See [`ScaledIdentity`](@ref) for the parameterization.
"""
homdiag(x) = StructuredReTerm{ScaledIdentity}(x)

"""
    homcs(term::RandomEffectsTerm)

Constrain the covariance matrix of the random effects in `term` to homogeneous
compound symmetry: a common variance and a common correlation.

See [`CompoundSymmetry`](@ref) for the parameterization.
"""
homcs(x) = StructuredReTerm{HomCS}(x)

"""
    cs(term::RandomEffectsTerm)

Constrain the covariance matrix of the random effects in `term` to
(heterogeneous) compound symmetry: per-coefficient variances and a common
correlation.

See [`CompoundSymmetry`](@ref) for the parameterization.
"""
cs(x) = StructuredReTerm{HetCS}(x)

# for schema extraction (from runtime-created wrappers)
StatsModels.terms(t::StructuredReTerm) = StatsModels.terms(t.term)
StatsModels.termvars(t::StructuredReTerm) = StatsModels.termvars(t.term)
StatsModels.degree(t::StructuredReTerm) = StatsModels.degree(t.term)

Base.show(io::IO, t::StructuredReTerm) = Base.show(io, MIME("text/plain"), t)

function StatsModels.apply_schema(
    t::StructuredReTerm{C}, sch::MultiSchema, Mod::Type{<:MixedModel}
) where {C}
    return StructuredReTerm{C}(apply_schema(t.term, sch, Mod))
end

function StatsModels.modelcols(t::StructuredReTerm{C}, d::NamedTuple) where {C}
    return structure!(modelcols(t.term, d), C)
end

function Base.getproperty(x::StructuredReTerm, s::Symbol)
    return s == :term ? getfield(x, s) : getproperty(x.term, s)
end

function _structured_inner(f::String, t::AbstractTerm)
    t isa RandomEffectsTerm ||
        throw(ArgumentError("covariance structure wrappers such as $f() cannot be nested"))
    return t
end

for (f, C) in (
    (:zerocorr, :ZeroCorrStruct),
    (:homdiag, :ScaledIdentity),
    (:homcs, :HomCS),
    (:cs, :HetCS),
)
    fstr = string(f)
    @eval begin
        # dirty rotten no good ugly hack: make sure wrapped ranef terms sort appropriately
        # cf https://github.com/JuliaStats/StatsModels.jl/blob/41b025409af03c0e019591ac6e817b22efbb4e17/src/terms.jl#L421-L422
        StatsModels.degree(t::FunctionTerm{typeof($f)}) = StatsModels.degree(only(t.args))

        function schema(t::FunctionTerm{typeof($f)}, data, hints::Dict{Symbol})
            return schema(only(t.args), data, hints)
        end

        function StatsModels.apply_schema(
            t::FunctionTerm{typeof($f)}, sch::MultiSchema, Mod::Type{<:MixedModel}
        )
            inner = _structured_inner($fstr, apply_schema(only(t.args), sch, Mod))
            return StructuredReTerm{$C}(inner)
        end

        function Base.show(io::IO, ::MIME"text/plain", t::StructuredReTerm{$C})
            # ranefterms already show with parens
            return print(io, $fstr, t.term)
        end
    end
end
