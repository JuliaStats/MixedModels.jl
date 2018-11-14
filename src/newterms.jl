struct ReMat{T,R,S} <: AbstractMatrix{T}
    trm::CategoricalTerm
    refs::Vector{R}
    z::Matrix{T}
    wtz::Matrix{T}
    wtzv::Base.ReinterpretArray{SVector{S,T}}
    λ::LowerTriangular{T,Matrix{T}}
    inds::Vector{Int}
    sparsemat::SparseMatrixCSC
end

function Base.size(A::ReMat)
    k, m = size(A.z)
    m, k * length(A.trm.contrasts.levels)
end

SparseArrays.sparse(A::ReMat) = A.sparsemat

Base.getindex(A::ReMat, i::Integer, j::Integer) = getindex(sparse(A), i, j)

Base.Matrix(A::ReMat) = Matrix(sparse(A))

"""
    nranef(A::AbstractMatrix)

Return the number of random effects represented by `A`.  Zero unless `A` is an `ReMat`.
""" 
nranef(A::ReMat{T,R,S}) where {T,R,S} = S * length(A.refs) 
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
    T = eltype(z)
    S = size(z, 1)
    grp = t.rhs
    m = reshape(1:abs2(S), (S, S))
    inds = sizehint!(Int[], (S * (S + 1)) >> 1)
    for j in 1:S, i in j:S
        push!(inds, m[i,j])
    end
    refs = getindex.(Ref(t.rhs.contrasts.invindex), d[grp.sym])
    R = eltype(refs)
    ReMat(grp, refs, z, z, reinterpret(SVector{S,eltype(z)}, vec(z)),
        LowerTriangular(Matrix{T}(I, S, S)), inds, 
        sparse(repeat(R.(1:length(refs)), inner=S), 
            R.(vec([(r - 1)*S + j for j in 1:S, r in refs])), vec(z)))
end
