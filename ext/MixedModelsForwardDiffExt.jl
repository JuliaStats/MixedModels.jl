module MixedModelsForwardDiffExt

using MixedModels
using MixedModels: AbstractReMat,
    block,
    BlockedSparse,
    cholUnblocked!,
    copyscaleinflate!,
    kp1choose2,
    LD,
    lmulΛ!,
    rmulΛ!,
    ssqdenom,
    UniformBlockDiagonal
using LinearAlgebra
using LinearAlgebra: PosDefException, copy_oftype
using SparseArrays
using Statistics

using ForwardDiff

function ForwardDiff.hessian(model::LinearMixedModel{T}, θσ::Vector{T}=[model.θ; model.σ]) where {T}
    return ForwardDiff.hessian(fd_deviance(model), θσ)
end

function ForwardDiff.gradient(model::LinearMixedModel{T}, θσ::Vector{T}=[model.θ; model.σ]) where {T}
    return ForwardDiff.gradient(fd_deviance(model), θσ)
end

#####
##### Evaluation of objective
#####

fd_deviance(model) = Base.Fix1(fd_deviance, model)

function fd_deviance(model::LinearMixedModel, θσ::AbstractVector{T}) where {T}
    σ² = θσ[end]^2
    θ = θσ[1:(end - 1)]
    dof = ssqdenom(model)

    # Extract and promote
    A, L, reterms = model.A, model.L, model.reterms
    AA = [copy_oftype(Ai, T) for Ai in A]
    LL = [copy_oftype(Li, T) for Li in L]
    RR = [copy_oftype(Ri, T) for Ri in reterms]

    # Update state with new θ
    fd_setθ!(RR, model.parmap, θ)
    fd_updateL!(AA, LL, RR)

    r² = fd_pwrss(LL)
    ld = fd_logdet(LL, RR, model.optsum.REML)

    return dof * log(2 * π * σ²) + ld + r² / σ²
end

function fd_setθ!(
    reterms::Vector{<:AbstractReMat},
    parmap::Vector{<:NTuple},
    θ::AbstractVector,
)
    length(θ) == length(parmap) || throw(DimensionMismatch())
    reind = 1
    λ = first(reterms).λ
    for (tv, tr) in zip(θ, parmap)
        tr1 = first(tr)
        if reind ≠ tr1
            reind = tr1
            λ = reterms[tr1].λ
        end
        λ[tr[2], tr[3]] = tv
    end
    return reterms
end

function fd_updateL!(A::Vector, L::Vector, reterms::Vector)
    k = length(reterms)
    copyto!(last(L), last(A))  # ensure the fixed-effects:response block is copied
    for j in eachindex(reterms) # pre- and post-multiply by Λ, add I to diagonal
        cj = reterms[j]
        diagind = kp1choose2(j)
        copyscaleinflate!(L[diagind], A[diagind], cj)
        for i in (j + 1):(k + 1)     # postmultiply column by Λ
            bij = block(i, j)
            rmulΛ!(copyto!(L[bij], A[bij]), cj)
        end
        for jj in 1:(j - 1)        # premultiply row by Λ'
            lmulΛ!(cj', L[block(j, jj)])
        end
    end
    for j in 1:(k + 1)             # blocked Cholesky
        Ljj = L[kp1choose2(j)]
        for jj in 1:(j - 1)
            fd_rankUpdate!(
                Hermitian(Ljj, :L),
                L[block(j, jj)],
                -one(eltype(Ljj)),
                one(eltype(Ljj)),
            )
        end
        fd_cholUnblocked!(Ljj, Val{:L})
        LjjT = isa(Ljj, Diagonal) ? Ljj : LowerTriangular(Ljj)
        for i in (j + 1):(k + 1)
            Lij = L[block(i, j)]
            for jj in 1:(j - 1)
                mul!(
                    Lij,
                    L[block(i, jj)],
                    L[block(j, jj)]',
                    -one(eltype(Lij)),
                    one(eltype(Lij)),
                )
            end
            rdiv!(Lij, LjjT')
        end
    end
    return nothing
end

fd_pwrss(L::Vector) = abs2(last(last(L)))

function fd_logdet(L::Vector, reterms::Vector{<:AbstractReMat}, REML::Bool)
    @inbounds s = sum(j -> LD(L[kp1choose2(j)]), axes(reterms, 1))
    if REML
        lastL = last(L)
        s += LD(lastL)        # this includes the log of sqrtpwrss
        s -= log(last(lastL)) # so we need to subtract it from the sum
    end
    return s + s  # multiply by 2 b/c the desired det is of the symmetric mat, not the factor
end

#####
##### Cholesky factorization
#####

function fd_cholUnblocked!(A::StridedMatrix, ::Type{Val{:L}})
    cholesky!(Hermitian(A, :L))
    return A
end
function fd_cholUnblocked!(D::UniformBlockDiagonal, ::Type{T}) where {T}
    Ddat = D.data
    for k in axes(Ddat, 3)
        fd_cholUnblocked!(view(Ddat,:,:,k), T)
    end
    return D
end
function fd_cholUnblocked!(A::Diagonal, ::Type{T}) where {T}
    A.diag .= sqrt.(A.diag)
    return A
end
fd_cholUnblocked!(A::AbstractMatrix, ::Type{T}) where {T} = cholUnblocked!(A, T)

#####
##### Rank update
#####

function fd_rankUpdate!(
    C::LinearAlgebra.HermOrSym{T,UniformBlockDiagonal{T}},
    A::StridedMatrix{T},
    α,
    β,
) where {T}
    Cdat = C.data.data
    LinearAlgebra.require_one_based_indexing(Cdat, A)
    isone(β) || rmul!(Cdat, β)
    blksize = size(Cdat, 1)

    for k in axes(Cdat, 3)
        ioffset = (k - 1) * blksize
        joffset = (k - 1) * blksize
        for i in axes(Cdat, 1), j in 1:i
            iind = ioffset + i
            jind = joffset + j
            AtAij = 0
            for idx in axes(A, 2)
                # because the second multiplicant is from A', swap index order
                AtAij += A[iind, idx] * A[jind, idx]
            end
            Cdat[i, j, k] += α * AtAij
        end
    end

    return C
end

function fd_rankUpdate!(
    C::LinearAlgebra.HermOrSym{T,UniformBlockDiagonal{T}},
    A::BlockedSparse{T,S},
    α,
    β,
) where {T,S}
    Ac = A.cscmat
    cp = Ac.colptr
    all(==(S), diff(cp)) ||
        throw(ArgumentError("Columns of A must have exactly $S nonzeros"))
    Cdat = C.data.data
    LinearAlgebra.require_one_based_indexing(Ac, Cdat)
    Cdat .*= β

    j, k, l = size(Cdat)
    S == j == k && div(Ac.m, S) == l ||
        throw(DimensionMismatch("div(A.cscmat.m, S) ≠ size(C.data.data, 3)"))
    nz = Ac.nzval
    rv = Ac.rowval

    @inbounds for j in axes(Ac, 2)
        nzr = nzrange(Ac, j)
        # BLAS.syr!('L', α, view(nz, nzr), view(Cdat, :, :, div(rv[last(nzr)], S)))
        _x = view(nz, nzr)
        view(Cdat,:,:,div(rv[last(nzr)], S)) .+= α .* _x .* _x'
    end

    return C
end

function fd_rankUpdate!(
    C::LinearAlgebra.HermOrSym{T,S},
    A::StridedMatrix{T},
    α,
    β,
) where {T,S}
    # BLAS.syrk!(C.uplo, 'N', T(α), A, T(β), C.data)
    C.data .*= β
    C.data .+= α .* A * A'
    return C
end

fd_rankUpdate!(C::AbstractMatrix, A::AbstractMatrix, α, β) = rankUpdate!(C, A, α, β)

end # module
