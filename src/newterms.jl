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

Base.size(A::ReMat) = size(A.sparsemat)

SparseArrays.sparse(A::ReMat) = A.sparsemat

Base.getindex(A::ReMat, i::Integer, j::Integer) = getindex(A.sparsemat, i, j)

"""
    nranef(A::AbstractMatrix)

Return the number of random effects represented by `A`.  Zero unless `A` is an `ReMat`.
""" 
nranef(A::ReMat) = size(A.sparsemat, 2)
nranef(A) = 0

function LinearAlgebra.mul!(C::Diagonal{T}, adjA::Adjoint{T,<:ReMat{T,R,1}},
    B::ReMat{T,R,1}) where {T,R}
    A = adjA.parent
    @assert A === B
    d = C.diag
    fill!(d, zero(T))
    @inbounds for (ri, Azi) in zip(A.refs, A.wtz)
        d[ri] += abs2(Azi)
    end
    C
end

function *(adjA::Adjoint{T,<:ReMat{T,R,1}}, B::ReMat{T,R,1}) where {T,R}
    A = adjA.parent
    A === B ? mul!(Diagonal(Vector{T}(undef, size(B, 2))), adjA, B) :
    sparse(Vector{Int32}(A.refs), Vector{Int32}(B.refs), vec(A.wtz .* B.wtz))
end

*(adjA::Adjoint{T,<:ReMat{T}}, B::ReMat{T}) where {T} = sparse(adjA.parent)'sparse(B)
*(adjA::Adjoint{T,<:VecOrMat{T}}, B::ReMat{T}) where {T} = adjA * sparse(B)


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
    invindex = t.rhs.contrasts.invindex
    refs = getindex.(Ref(invindex), d[grp.sym])
    R = eltype(refs)
    ReMat(grp, refs, z, z, reinterpret(SVector{S,eltype(z)}, vec(z)),
        LowerTriangular(Matrix{T}(I, S, S)), inds, 
        sparse(repeat(R.(1:length(refs)), inner=S), 
            R.(vec([(r - 1)*S + j for j in 1:S, r in refs])), vec(z),
            size(z,2), S * length(invindex)))
end
