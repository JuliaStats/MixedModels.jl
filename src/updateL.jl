"""
    scaleinflate!(L::AbstractMatrix, Λ::ReMat)

Overwrite `L` with `Λ'LΛ + I`
"""
function scaleinflate! end

function scaleinflate!(Ljj::Diagonal{T}, Λj::ReMat{T,1}) where {T}
    Ljjd = Ljj.diag
    Ljjd .= Ljjd .* abs2(only(Λj.λ)) .+ one(T)
    Ljj
end

function scaleinflate!(Ljj::Matrix{T}, Λj::ReMat{T,1}) where {T}
    lambsq = abs2(only(Λj.λ))
    @inbounds for i in diagind(Ljj)
        Ljj[i] *= lambsq
        Ljj[i] += one(T)
    end
    Ljj
end

function scaleinflate!(Ljj::UniformBlockDiagonal{T}, Λj::ReMat{T,S}) where {T,S}
    A = Ljj.data
    m, n, l = size(A)
    m == n == S || throw(DimensionMismatch())
    λ = Λj.λ
    for f in 1:l
        lmul!(λ', rmul!(view(A, :, :, f), λ))
        for k in 1:S
            A[k, k, f] += one(T)
        end
    end
    Ljj
end

function scaleinflate!(Ljj::Matrix{T}, Λj::ReMat{T,S}) where{T,S}
    n = checksquare(Ljj)
    q, r = divrem(n, S)
    iszero(r) || throw(DimensionMismatch("size(Ljj, 1) is not a multiple of S"))
    λ = Λj.λ
    offset = 0
    @inbounds for k in 1:q
        inds = (offset + 1):(offset + S)
        lmul!(λ', rmul!(view(Ljj, inds, inds), λ))
        offset += S
    end
    for k in diagind(Ljj)
        Ljj[k] += 1
    end
    Ljj
end

function lmulΛ!(adjA::Adjoint{T,<:LowerTriangular{T,Matrix{T}}}, B::UniformBlockDiagonal{T}) where {T}
    Bd = B.data
    for k in axes(Bd, 3)
        lmul!(adjA, view(Bd, :, :, k))
    end
    B
end

function lmulΛ!(adjA::Adjoint{T,<:LowerTriangular{T}}, B::Diagonal{T}) where {T}
    lmul!(only(adjA.parent.data), B.diag)
    B
end

function rmulΛ!(A::UniformBlockDiagonal{T}, B::LowerTriangular{T}) where {T}
    Ad = A.data
    for k in axes(Ad, 3)
        rmul!(view(Ad, :, :, k), B)
    end
    A
end

symmetrize!(A::UniformBlockDiagonal) = byface(symmetrize!, A)

inflatediag!(A::UniformBlockDiagonal) = byface(inflatediag!, A)

"""
    diagratio(Ldot::AbstractMatrix{T}, L::AbstractMatrix{T}) where {T<:Number}

Return `sum(diag(Ldot) ./ diag(L))`
"""
function diagratio(Ldot::UniformBlockDiagonal{T}, L::UniformBlockDiagonal{T}) where {T}
    Ldotdat, Ldat = Ldot.data, L.data
    size(Ldotdat) == size(Ldat) || throw(DimensionMismatch())
    s = zero(T) / one(T)
    for k in axes(Ldat, 3)
        for j in axes(Ldat, 2)
            s += Ldotdat[j, j, k] / Ldat[j, j, k]
        end
    end
    s
end

function diagratio(Ldot::Diagonal{T}, L::Diagonal{T}) where {T}
    sum(Ldot.diag ./ L.diag)
end

function diagratio(Ldot::Matrix{T}, L::Matrix{T}) where {T}
    size(Ldot) == size(L) || throw(DimensionMismatch())
    sum(Ldot[k] / L[k] for k in diagind(L))
end

function fg!(g::AbstractVector{T}, m::LinearMixedModel{T}) where {T}
    L, Ldot, parmap = m.L, m.Ldot, m.parmap
    n = size(first(m.allterms), 1)
    k = BlockArrays.blocksize(L, 1)
    nre = length(m.reterms)
    initializeΩdot!(m)
    length(Ldot) == length(g) || throw(DimensionMismatch())
    for (i, ld) in enumerate(Ldot)
        updateLdot!(ld, L, first(parmap[i]))
        g[i] = 2*(sum(diagratio(ld[Block(j,j)], L[Block(j,j)]) for j in 1:nre) + 
            n * last(ld) / last(L))
    end
    objective(m)
end

#=
"""
    copycol!(A, B, i, j, symm::Bool=false)

Zero `A` then copy the `i`'th column of `B` to the `j`'th column of `A`.  Optionally symmetrize A by adding its adjoint.
""" 
function copycol!(A::AbstractMatrix{T}, B::AbstractMatrix{T}, i, j, symm::Bool=false) where{T}
    fill!(A, zero(T))
    copyto!(view(A, :, j), view(B, :, i))
    symm && symmetrize!(A)
    A
end
=#

function inflatediag!(A::AbstractMatrix)
    @inbounds for k in diagind(A)
        A[k] += true
    end
    A
end

function rmulΛdot!(Ld::Diagonal, L::Diagonal, trp, K, symm::Bool)
    Lddiag, Ldiag = Ld.diag, L.diag
    if !((trp[2] == trp[3] == K == 1) && symm)
        throw(ArgumentError("For Diagonal must have trp[2] == trp[3] == K == 1 and symm"))
    end
    if length(Lddiag) ≠ length(Ldiag)
        throw(DimensionMismatch("length(Ld.diag) = $(length(Lddiag)) ≠ $(length(Ldiag)) = length(L.diag)"))
    end
    @inbounds for i in axes(Ldiag, 1)
        src = Ldiag[i]
        Lddiag[i] = src + src
    end
    Ld
end

function rmulΛdot!(Ld::UniformBlockDiagonal, L::UniformBlockDiagonal, trp, K, symm::Bool)
    @assert symm "UniformBlockDiagonal only makes sense for a diagonal block"
    Lddat, Ldat = Ld.data, L.data
    if size(Lddat) ≠ size(Ldat)
        throw(DimensionMismatch("size(Ld.data) = $(size(Lddat)) ≠ $(size(L.data)) = size(L.data)"))
    end
    for k in axes(Ldat, 3)
        dest = view(Lddat, :, :, k)
        j = trp[3]     # destination row/column
        for (i, src) in enumerate(view(Ldat, :, trp[2], k))
            if i < j
                dest[j, i] = src
            elseif i == j
                dest[j, j] = src + src
            else
                dest[i, j] = src
            end
        end
    end
    Ld
end

function rmulΛdot!(Ld::Matrix, L::Matrix, trp, K, symm::Bool)
    size(Ld) == size(L) || throw(DimensionMismatch())
    d, r = divrem(size(L, 2), K)
    iszero(r) || throw(DimensionMismatch("size(L, 2) == $(size(L,2)) is not a multiple of K = $K"))
    srccol, destcol = trp[2], trp[3]
    for k in 1:d
        offset = (k-1)*K
        copyto!(view(Ld, :, offset + destcol), view(L, :, offset + srccol))
    end
    symm ? symmetrize!(Ld) : Ld
end

function scalecol(m::LinearMixedModel)
    A, L, Ldot, trms = m.A, m.L, m.Ldot, m.reterms
    offset = 0
    for (j, trm) in enumerate(trms)
        Ljj = lmulΛ!(trm', copyto!(L[Block(j,j)], A[Block(j,j)]))
        for k in axes(trm.inds, 1)
            symmetrize!(rmulΛ!(copyto!(Ldot[offset + k][Block(j,j)], Ljj), λdot(trm, k)))
        end
        inflatediag!(rmulΛ!(Ljj, trm))
        offset += length(trm.inds)
    end
    m
end
#=
"""
    skewscale!(Ljj, re::ReMat, ind::Integer)

Overwrite `Ljj` by `symmetrize!(λdot(re, ind)'Ljj*re.λ)`
"""
function skewscale! end

function skewscale!(Ljj::Diagonal{T}, re::ReMat{T,1}, ind::Integer) where {T}
    @assert isone(ind)
    Ljj.diag .*= (first(re.λ.data) * T(2))
    Ljj
end

function skewscale!(Ljj::Matrix{T}, re::ReMat{T,1}, ind::Integer) where {T}
    @assert isone(ind)
    Ljj .*= first(re.λ.data)
    symmetrize!(Ljj)
end

function skewscale!(Ljj::UniformBlockDiagonal{T}, re::ReMat{T,S}, ind::Integer) where {T,S}
    λd = λdot(re, ind)
    λ = re.λ
    Ljjd = Ljj.data
    for k in axes(Ljjd, 3)
        symmetrize!(lmul!(λd', rmul!(view(Ljjd, :, :, k), λ)))
    end
    Ljj
end
=#
"""
    updateL!(m::LinearMixedModel)

Update the blocked lower Cholesky factor, `m.L`, from `m.A` and `m.reterms` (used for λ only)

This is the crucial step in evaluating the objective, given a new parameter value.
"""
function updateL!(m::LinearMixedModel{T}) where {T}
    A, L = m.A, m.L
    k = BlockArrays.blocksize(A, 2)
    for j = 1:k                         # copy lower triangle of A to L
        for i = j:k
            copyto!(L[Block(i, j)], A[Block(i, j)])
        end
    end
    for (j, cj) in enumerate(m.reterms)  # pre- and post-multiply by Λ, add I to diagonal
        scaleinflate!(L[Block(j, j)], cj)
        for i = (j+1):k         # postmultiply column by Λ
            rmulΛ!(L[Block(i, j)], cj)
        end
        for jj = 1:(j-1)        # premultiply row by Λ'
            lmulΛ!(cj', L[Block(j, jj)])
        end
    end
    cholBlocked!(L)
    m
end

"""
    cholBlocked!(L::BlockArray)

Overwrite the lower triangle of `L` with its left Cholesky factor
"""
function cholBlocked!(L::BlockMatrix{T}) where {T}
    k = BlockArrays.blocksize(L, 2)
    for j = 1:k                         # blocked Cholesky
        Ljj = L[Block(j, j)]
        for jj = 1:(j-1)
            rankUpdate!(Hermitian(Ljj, :L), L[Block(j, jj)], -one(T))
        end
        cholUnblocked!(Ljj, Val{:L})
        LjjT = isa(Ljj, Diagonal) ? Ljj : LowerTriangular(Ljj)
        for i = (j+1):k
            Lij = L[Block(i, j)]
            for jj = 1:(j-1)
                mul!(Lij, L[Block(i, jj)], L[Block(j, jj)]', -one(T), one(T))
            end
            rdiv!(Lij, LjjT')
        end
    end
    L
end

function initializeΩdot!(m::LinearMixedModel)
    A, L, Ldot, parmap, reterms = m.A, m.L, m.Ldot, m.parmap, m.reterms
    k = BlockArrays.blocksize(A, 2)
    nre = length(reterms)
    for j = 1:k             # copy lower triangle of A to L and zero each element of Ldot
        for i = j:k
            copyto!(L[Block(i, j)], A[Block(i, j)])
            for ldot in Ldot
                zero!(ldot[Block(i, j)])
            end
        end
    end
    for (i, ci) in enumerate(reterms)  # left multiply L by Λ
        for j in 1:i
            lmulΛ!(ci', L[Block(i), Block(j)])
        end
    end
    for (j, cj) in enumerate(reterms)
        K = size(cj.λ, 2)
        for (trp, ldot) in zip(parmap, Ldot)
            if j == trp[1]
                for i in j:k
                    rmulΛdot!(ldot[Block(i,j)], L[Block(i,j)], trp, K, i == j)
                end
            end
        end
        Ljj = inflatediag!(rmulΛ!(L[Block(j, j)], cj))
        for i in (j+1):k
            rmulΛ!(L[Block(i,j)], cj)
        end
    end
    cholBlocked!(L)
    m
end

function LinearAlgebra.mul!(
    C::Matrix{T}, 
    A::Matrix{T}, 
    adjB::Adjoint{T,LowerTriangular{T,UniformBlockDiagonal{T}}},
    alpha::Number, 
    beta::Number,
) where{T}
    Bdat = adjB.parent.data.data
    l, m, n = size(Bdat)
    offset = 0
    for k in axes(Bdat, 3)
        colblock = (offset + 1):(offset + m)
        mul!(
            view(C, :, colblock), 
            view(A, :, colblock), 
            LowerTriangular(view(Bdat, :, :, k))',
            alpha,
            beta,
            )
        offset += m
    end
    C
end

function LinearAlgebra.rdiv!(A::Matrix{T}, adjB::Adjoint{LowerTriangular{T,UniformBlockDiagonal{T}}}) where {T}
    Bdat = adjB.parent.data.data
    l, m, n = size(Bdat)
    offset = 0
    for k in axes(Bdat, 3)
        colblock = (offset + 1):(offset + m)
        rdiv!(view(A, :, colblock), LowerTriangular(view(Bdat, :, :, k))')
        offset += m
    end
    A
end

function rank2Update!(C::Hermitian{T,Matrix{T}}, A::Matrix{T}, B::Matrix{T}, alpha::Number, beta::Number) where {T}
    BLAS.syr2k!(C.uplo, 'N', T(alpha), A, B, T(beta), C.data)
end

"""
    updateLdot!(m::LinearMixedModel)

Update the blocked lower Cholesky factor, `m.L`, and the sensitivites, `m.Ldot`, from `m.A` and `m.reterms`

This is the crucial step in evaluating the objective and gradient for a new parameter value.
"""
function updateLdot!(Ldot::BlockArray{T}, L::BlockArray{T}, blk) where {T}
    K = BlockArrays.blocksize(L, 2)
    for k in blk:K
        Ldotkk = Ldot[Block(k, k)]
        for j in 1:(k - 1)
            rank2Update!(Hermitian(Ldotkk, :L), Ldot[Block(k, j)], L[Block(k, j)], -one(T), one(T))
            for i in (k + 1):K
                mul!(
                    mul!(Ldot[Block(i, k)], Ldot[Block(i, j)], L[Block(k, j)]', -one(T), one(T)),
                    L[Block(i, j)],
                    Ldot[Block(k, j)]',
                    -one(T),
                    one(T)
                )
            end
        end
        Lkk = L[Block(k, k)]
        chol_unblocked_fwd!(Ldotkk, Lkk)
        for i in (k + 1):K
            rdiv!(
                mul!(Ldot[Block(i, k)], L[Block(i, k)], LowerTriangular(Ldotkk)', -one(T), one(T)),
                LowerTriangular(Lkk)',
            )
        end
    end
    Ldot
end

zero!(A::AbstractArray) = fill!(A, false)
zero!(A::BlockedSparse) = zero!(A.cscmat.nzval)
zero!(A::SparseMatrixCSC) = zero!(A.nzval)
zero!(A::UniformBlockDiagonal) = zero!(A.data)
