struct ReMat{T,R,S} <: AbstractMatrix{T}
    trm::CategoricalTerm
    refs::Vector{R}
    z::Matrix{T}
    wtz::Matrix{T}
    wtzv::Base.ReinterpretArray{SVector{S,T}}
end

function Base.size(A::ReMat)
    k, m = size(A.z)
    m, k * length(A.trm.contrasts.levels)
end

SparseArrays.sparse(A::ReMat{T,R,1}) where {T,R} =
    sparse(R.(1:length(A.refs)), A.refs, vec(A.z))

function SparseArrays.sparse(A::ReMat{T,R,S}) where {T,R,S}
    I = repeat(R.(1:length(A.refs)), inner=S)
    J = R.(vec([(r - 1)*S + j for j in 1:S, r in A.refs]))
    sparse(I, J, vec(A.z))
end

Base.getindex(A::ReMat, i::Integer, j::Integer) = getindex(sparse(A), i, j)

Base.Matrix(A::ReMat) = Matrix(sparse(A))

"""
    nranef(A::AbstractMatrix)

Return the number of random effects represented by `A`.  Zero unless `A` is an `ReMat`.
""" 
nranef(A::ReMat{T,R,S}) where {T,R,S} = S*length(A.refs) 
nranef(A) = 0

*(A::Adjoint{T,ReMat{T}}, B::ReMat{T}) where {T} = sparse(A)'sparse(B)
*(A::Adjoint{T,Matrix{T}}, B::ReMat{T}) where {T} = A'sparse(B)

abstract type MixedModel{T} <: StatsModels.RegressionModel end # model with fixed and random effects

struct RandomEffectsTerm <: AbstractTerm
    lhs::TermOrTerms
    rhs::CategoricalTerm
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
    ReMat(grp, getindex.(Ref(t.rhs.contrasts.invindex), d[grp.sym]), z, z,
        reinterpret(SVector{k,eltype(z)}, vec(z)))
end
