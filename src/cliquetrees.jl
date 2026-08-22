struct ChordalWorkspace{T}
    blocks::Vector{AbstractMatrix{T}}
    indices::Vector{Vector{Int}}
end

_nonzeros(A::Diagonal) = A.diag
_nonzeros(A::Matrix) = vec(A)
_nonzeros(A::UniformBlockDiagonal) = vec(A.data)
_nonzeros(A::BlockedSparse) = nonzeros(A.cscmat)
_nonzeros(A::SparseMatrixCSC) = nonzeros(A)

_nnz(A::Diagonal) = length(A.diag)
_nnz(A::Matrix) = length(A)
_nnz(A::UniformBlockDiagonal) = length(A.data)
_nnz(A::BlockedSparse) = nnz(A.cscmat)
_nnz(A::SparseMatrixCSC) = nnz(A)

_similar(A::Diagonal{T}) where {T} = Diagonal(similar(A.diag))
_similar(A::Matrix{T}) where {T} = similar(A)
_similar(A::UniformBlockDiagonal{T}) where {T} = UniformBlockDiagonal(similar(A.data))
_similar(A::BlockedSparse{T}) where {T} = BlockedSparse(similar(A.cscmat), A.nzasmat, A.colblkptr)
_similar(A::SparseMatrixCSC{T}) where {T} = similar(A)

function _csc(blocks::Vector{<:AbstractMatrix{T}}, reterms::Vector{<:AbstractReMat{T}}) where {T}
    nblkcol = length(reterms) + 1
    nblkptr = length(blocks)
    nfixcol = size(blocks[kp1choose2(nblkcol)], 1)
    ncol = nfixcol
    nptr = 0

    indices = Vector{Vector{Int}}(undef, nblkptr)

    for R in reterms
        ncol += size(R.λ, 1) * nlevs(R)
    end

    for (i, B) in enumerate(blocks)
        nptr += n = _nnz(B)
        indices[i] = Vector{Int}(undef, n)
    end

    colptr = Vector{Int}(undef, ncol + 1)
    rowval = Vector{Int}(undef, nptr)
    nzval = Vector{T}(undef, nptr)
    col = 1; colptr[col] = ptr = 1
    noffcol = 0

    for blkcol in 1:nblkcol
        if blkcol < nblkcol
            R = reterms[blkcol]; nloccol = size(R.λ, 1) * nlevs(R)
        else
            nloccol = nfixcol
        end

        for loccol in 1:nloccol
            noffrow = noffcol

            for blkrow in blkcol:nblkcol
                blkptr = block(blkrow, blkcol)

                ptr = _addcol!(rowval, nzval, indices[blkptr], blocks[blkptr], loccol, noffrow, ptr)

                if blkrow < nblkcol
                    noffrow += size(reterms[blkrow].λ, 1) * nlevs(reterms[blkrow])
                else
                    noffrow += nfixcol
                end
            end

            col += 1; colptr[col] = ptr
        end

        noffcol += nloccol
    end

    S = SparseMatrixCSC(ncol, ncol, colptr, rowval, nzval)
    return S, indices
end

function _addcol!(
        rowval::Vector{Int},
        nzval::Vector{T},
        ind::Vector{Int},
        A::Diagonal{T},
        loccol::Int,
        noffrow::Int,
        ptr::Int,
    ) where {T}
    rowval[ptr] = noffrow + loccol
    nzval[ptr] = A.diag[loccol]
    ind[loccol] = ptr
    return ptr + 1
end

function _addcol!(
        rowval::Vector{Int},
        nzval::Vector{T},
        ind::Vector{Int},
        A::Matrix{T},
        loccol::Int,
        noffrow::Int,
        ptr::Int,
    ) where {T}
    for locrow in axes(A, 1)
        rowval[ptr] = noffrow + locrow
        nzval[ptr] = A[locrow, loccol]
        ind[(loccol - 1) * size(A, 1) + locrow] = ptr
        ptr += 1
    end

    return ptr
end

function _addcol!(
    rowval::Vector{Int},
    nzval::Vector{T},
    ind::Vector{Int},
    A::UniformBlockDiagonal{T},
    loccol::Int,
    noffrow::Int,
    ptr::Int,
) where {T}
    nsubrow, nsubcol, nlevel = size(A.data)
    level = (loccol - 1) ÷ nsubcol + 1
    subcol = (loccol - 1) % nsubcol + 1
    noffrow += (level - 1) * nsubrow

    for locrow in 1:nsubrow
        rowval[ptr] = noffrow + locrow
        nzval[ptr] = A.data[locrow, subcol, level]
        ind[(level - 1) * nsubrow * nsubcol + (subcol - 1) * nsubrow + locrow] = ptr
        ptr += 1
    end

    return ptr
end

function _addcol!(
    rowval::Vector{Int},
    nzval::Vector{T},
    ind::Vector{Int},
    A::BlockedSparse{T},
    loccol::Int,
    noffrow::Int,
    ptr::Int,
) where {T}
    return _addcol!(rowval, nzval, ind, A.cscmat, loccol, noffrow, ptr)
end

function _addcol!(
    rowval::Vector{Int},
    nzval::Vector{T},
    ind::Vector{Int},
    A::SparseMatrixCSC{T},
    loccol::Int,
    noffrow::Int,
    ptr::Int,
) where {T}
    for p in nzrange(A, loccol)
        locrow = rowvals(A)[p]
        rowval[ptr] = noffrow + locrow
        nzval[ptr] = nonzeros(A)[p]
        ind[p] = ptr; ptr += 1
    end

    return ptr
end

function _addblk!(F::ChordalCholesky, A::AbstractMatrix{T}, P::Vector{Int}) where {T}
    src = _nonzeros(A)

    for k in eachindex(P)
        iszero(P[k]) || setflatindex!(F, src[k], P[k])
    end

    return
end

function _init(M::LinearMixedModel{T}; kw...) where {T}
    # get sparsity pattern of `M`
    S, indices = _csc(M.A, M.reterms)

    # construct clique tree
    perm, tree = cliquetree(Symmetric(S, :L); kw...)
    invp =  invperm(perm)

    # add fixed-effects columns to `clique`
    nfixcol = size(M.A[end], 1)
    clique = invp[end - nfixcol + 1:end]

    # rotate clique tree so that fixed-effects columns
    # are eliminated last
    permute!(perm, cliquetree!(tree, clique))

    # construct uninitialized Cholesky factor
    F = ChordalCholesky{:L, T}(perm, ChordalSymbolic(tree))

    # construct mapping from blocks into Cholesky factor
    P = flatindices(F, Symmetric(S, :L))

    for blkptr in eachindex(indices)
        index = indices[blkptr]

        for i in eachindex(index)
            index[i] = P[index[i]]
        end
    end

    # construct workspace
    W = ChordalWorkspace{T}(map(_similar, M.A), indices)
    return F, W
end

function _logdet(M::LinearMixedModel, F::ChordalCholesky)
    #
    # log|Ω_ZZ| = 2 Σ_j log|L_jj|   where  L L' = Ω_ZZ
    #
    q = _nzz(M, F)
    D = view(diag(F.L), 1:q)
    return 2sum(log ∘ abs, D)
end

function _pwrss(F::ChordalCholesky)
    D, _ = diagblock(F, nfr(F))
    return D[end, end]^2
end

function _transform!(
        M::LinearMixedModel{T},
        F::ChordalCholesky{UPLO, T},
        W::ChordalWorkspace{T},
    ) where {UPLO, T}
    # apply Λ-transform to M, writing the result to `W.blocks`.
    for k in eachindex(M.reterms)
        R = M.reterms[k]
        blkptr = kp1choose2(k)
        copyscaleinflate!(W.blocks[blkptr], M.A[blkptr], R)

        for i in k + 1:length(M.reterms) + 1
            blkptr = block(i, k)
            rmulΛ!(copyto!(W.blocks[blkptr], M.A[blkptr]), R)
        end

        for j in 1:k - 1
            blkptr = block(k, j)
            lmulΛ!(R', W.blocks[blkptr])
        end
    end

    blkptr = kp1choose2(length(M.reterms) + 1)
    copyto!(W.blocks[blkptr], M.A[blkptr])

    # write `W.blocks` directly to `F`
    fill!(F, zero(T))

    for blkptr in eachindex(W.blocks, W.indices)
        _addblk!(F, W.blocks[blkptr], W.indices[blkptr])
    end

    return F
end

function _nzz(M::LinearMixedModel, F::ChordalCholesky)
    return size(F.L, 1) - size(M.A[end], 1)
end

function _prepare_selinv_ZZ!(F::ChordalCholesky{UPLO, T}, q::Integer) where {UPLO, T}
    m = nfr(F)

    for j in 1:m - 1
        L, sep = offdblock(F, j)
        i = searchsortedfirst(sep, q + 1)
        L[i:end, :] .= zero(T)
    end

    D, res = diagblock(F, m)
    n = findfirst(==(q), res)
    D[n + 1:end, :] .= zero(T)
    D[diagind(D)[n + 1:end]] .= one(T)

    return F
end

function _trace_diag_block end

# Scalar RE (S=1)
function _trace_diag_block(
        F::ChordalCholesky{UPLO, T},
        ind::Vector{Int},
        A::Diagonal{T},
        Λ::ReMat{T, 1},
    ) where {UPLO, T}
    #
    # Ω_kk = λ² A + I   →   ∂Ω_kk/∂λ = 2λA   →   tr(Ω⁻¹ ∂Ω/∂λ) = Σ_j  Ω⁻¹_jj · 2λ · A_jj
    #
    λ = only(Λ.λ.data)
    out = zero(T)

    for (p, Ajj) in zip(ind, A.diag)
        if !iszero(p)
            invΩjj = getflatindex(F, p)
            out += invΩjj * 2λ * Ajj
        end
    end

    return out
end

# Vector RE (S>1): Λ is lower triangular, need (i,j) position
function _trace_diag_block(
    F::ChordalCholesky{UPLO, T},
    ind::Vector{Int},
    A::UniformBlockDiagonal{T},
    Λ::ReMat{T, S},
    pi::Int,
    pj::Int,
) where {UPLO, T, S}
    #
    # Ωkk = Λ'AΛ + I   →   ∂Ωkk/∂Λij = Eji A Λ + Λ'A Eij
    #
    # tr(Ωinv ∂Ω/∂Λij) = Σc Ωinv_pi,c · (AΛ)pj,c  +  Σr (Λ'A)r,pi · Ωinv_r,pj
    #
    λ = Λ.λ
    Adata = A.data
    nsubrow, nsubcol, nlevel = size(Adata)
    out = zero(T)

    for level in 1:nlevel
        Alev = view(Adata, :, :, level)

        # First term: Σc Ωinv_pi,c · (AΛ)pj,c
        for subcol in 1:nsubcol
            p = ind[(level - 1) * nsubrow * nsubcol + (subcol - 1) * nsubrow + pi]

            if !iszero(p)
                Aλ = zero(T)
                for k in 1:nsubcol
                    Aλ += Alev[pj, k] * λ[k, subcol]
                end
                invΩ = getflatindex(F, p)
                out += invΩ * Aλ
            end
        end

        # Second term: Σr (Λ'A)r,pi · Ωinv_r,pj
        for subrow in 1:nsubrow
            p = ind[(level - 1) * nsubrow * nsubcol + (pj - 1) * nsubrow + subrow]

            if !iszero(p)
                λA = zero(T)
                for k in 1:nsubrow
                    λA += λ[k, subrow] * Alev[k, pi]
                end
                invΩ = getflatindex(F, p)
                out += invΩ * λA
            end
        end
    end

    return out
end

function _trace_offdiag_block end

# Off-diagonal with scalar REs
function _trace_offdiag_block(
    F::ChordalCholesky{UPLO, T},
    ind::Vector{Int},
    A::AbstractMatrix{T},
    Λrow::ReMat{T, 1},
    Λcol::ReMat{T, 1},
    iscolpar::Bool,
) where {UPLO, T}
    #
    # Ωrk = λr A λk   →   ∂Ωrk/∂λk = λr A   or   ∂Ωrk/∂λr = A λk
    #
    # tr(Ωinv ∂Ω/∂λ) = 2 · Σij Ωinv_ij · scale · Aij   (×2 for symmetry)
    #
    λrow = only(Λrow.λ.data)
    λcol = only(Λcol.λ.data)
    scale = iscolpar ? λrow : λcol
    out = zero(T)

    for (p, Aij) in zip(ind, _nonzeros(A))
        if !iszero(p)
            invΩij = getflatindex(F, p)
            out += invΩij * scale * Aij
        end
    end

    return 2 * out
end

# Off-diagonal with vector REs - parameter in column term
function _trace_offdiag_block(
    F::ChordalCholesky{UPLO, T},
    ind::Vector{Int},
    A::BlockedSparse{T},
    Λrow::ReMat{T, Srow},
    Λcol::ReMat{T, Scol},
    pi::Int,
    pj::Int,
    iscolpar::Bool,
) where {UPLO, T, Srow, Scol}
    #
    # Ωrk = Λr' A Λk   →   ∂Ω/∂Λkij = Λr' A Eij   or   ∂Ω/∂Λrij = Eji A Λk
    #
    # tr(Ωinv ∂Ω) = 2 · Σij Ωinv_ij · Aij   (×2 for symmetry)
    #
    out = zero(T)

    for (p, Aij) in zip(ind, nonzeros(A.cscmat))
        if !iszero(p)
            invΩij = getflatindex(F, p)
            out += invΩij * Aij
        end
    end

    return 2 * out
end

function _logdet_gradient(
    M::LinearMixedModel{T},
    F::ChordalCholesky{UPLO, T},
    W::ChordalWorkspace{T},
) where {UPLO, T}
    g = zeros(T, length(M.parmap))
    #
    # d log|ΩZZ| / dθp = tr(ΩZZinv · dΩZZ/dθp)
    #
    # Sum over diagonal block (k,k) and off-diagonal blocks (r,k), (k,j)
    #
    parmap = M.parmap
    reterms = M.reterms
    nre = length(reterms)

    for (p, pm) in enumerate(parmap)
        k, pi, pj = pm  # term index, row, col in Λk
        result = zero(T)

        # Diagonal block (k,k): tr(ΩZZinv dΩkk)
        blkptr = kp1choose2(k)
        Λk = reterms[k]

        if Λk isa ReMat{T, 1}
            result += _trace_diag_block(F, W.indices[blkptr], M.A[blkptr], Λk)
        else
            result += _trace_diag_block(F, W.indices[blkptr], M.A[blkptr], Λk, pi, pj)
        end

        # Off-diagonal (r,k) for r > k: tr(ΩZZinv dΩrk) (×2 for symmetry)
        for r in k + 1:nre
            blkptr = block(r, k)
            Λr = reterms[r]

            if Λk isa ReMat{T, 1} && Λr isa ReMat{T, 1}
                result += _trace_offdiag_block(F, W.indices[blkptr], M.A[blkptr], Λr, Λk, true)
            else
                result += _trace_offdiag_block(F, W.indices[blkptr], M.A[blkptr], Λr, Λk, pi, pj, true)
            end
        end

        # Off-diagonal (k,j) for j < k: tr(ΩZZinv dΩkj) (×2 for symmetry)
        for j in 1:k - 1
            blkptr = block(k, j)
            Λj = reterms[j]

            if Λk isa ReMat{T, 1} && Λj isa ReMat{T, 1}
                result += _trace_offdiag_block(F, W.indices[blkptr], M.A[blkptr], Λk, Λj, false)
            else
                result += _trace_offdiag_block(F, W.indices[blkptr], M.A[blkptr], Λk, Λj, pi, pj, false)
            end
        end

        # Note: FE-RE blocks (nre+1,k) don't contribute since X'Z has no θ dependency

        g[p] = result
    end

    return g
end

function _quadform_Ωdot(
    cZZ::AbstractVector{T},
    cFE::AbstractVector{T},
    M::LinearMixedModel{T},
    W::ChordalWorkspace{T},
    p::Int,
) where {T}
    #
    # c'∂Ωc = cZZ'∂ΩZZ cZZ + 2·cZZ'∂ΩZF cFE   (FF term is zero)
    #
    parmap = M.parmap
    reterms = M.reterms
    nre = length(reterms)

    k, pi, pj = parmap[p]
    Λk = reterms[k]
    out = zero(T)

    # Offset to term k in cZZ
    noffk = 0
    for blk in 1:k - 1
        noffk += size(reterms[blk].λ, 1) * nlevs(reterms[blk])
    end

    #
    # Diagonal: ck' ∂Ωkk ck
    #
    blkptr = kp1choose2(k)
    nk = size(Λk.λ, 1) * nlevs(Λk)
    ck = view(cZZ, (noffk + 1):(noffk + nk))
    out += _quadform_diag_block(ck, M.A[blkptr], Λk, pi, pj)

    #
    # Off-diagonal (r,k) for r > k:  2·cr'∂Ωrk ck
    #
    noffr = noffk + nk

    for r in k + 1:nre
        Λr = reterms[r]
        nr = size(Λr.λ, 1) * nlevs(Λr)
        cr = view(cZZ, (noffr + 1):(noffr + nr))

        blkptr = block(r, k)
        out += 2 * _quadform_offdiag_block(cr, ck, M.A[blkptr], Λr, Λk, pi, pj, true)

        noffr += nr
    end

    #
    # Off-diagonal (k,j) for j < k:  2·ck'∂Ωkj cj
    #
    noffj = 0

    for j in 1:k - 1
        Λj = reterms[j]
        nj = size(Λj.λ, 1) * nlevs(Λj)
        cj = view(cZZ, (noffj + 1):(noffj + nj))

        blkptr = block(k, j)
        out += 2 * _quadform_offdiag_block(ck, cj, M.A[blkptr], Λk, Λj, pi, pj, false)

        noffj += nj
    end

    #
    # FE cross-term:  2·ck' ∂ΩZFk cFE
    #
    blkptrfe = block(nre + 1, k)
    out += 2 * _quadform_fe_cross(ck, cFE, M.A[blkptrfe], Λk, pi, pj)

    return out
end

# Quadratic form for diagonal block: c' dΩ c
function _quadform_diag_block(
    c::AbstractVector{T},
    A::Diagonal{T},
    Λ::ReMat{T, 1},
    pi::Int,
    pj::Int,
) where {T}
    #
    # Ωkk = λ²A + I   →   ∂Ωkk/∂λ = 2λA   →   c'(∂Ω)c = Σj cj² · 2λ · Ajj
    #
    λ = only(Λ.λ.data)
    out = zero(T)

    for (cj, Ajj) in zip(c, A.diag)
        out += cj * 2λ * Ajj * cj
    end

    return out
end

function _quadform_diag_block(
    c::AbstractVector{T},
    A::UniformBlockDiagonal{T},
    Λ::ReMat{T, S},
    pi::Int,
    pj::Int,
) where {T, S}
    #
    # Ωkk = Λ'AΛ + I   →   ∂Ω/∂Λij = Eji A Λ + Λ'A Eij
    #
    # c'(∂Ω)c = cpi · (AΛ)pj · c  +  c · (Λ'A)pi · cpj
    #
    λ = Λ.λ
    Adata = A.data
    nsubrow, nsubcol, nlevel = size(Adata)
    out = zero(T)

    Aλrow = zeros(T, nsubcol)
    λAcol = zeros(T, nsubrow)

    for level in 1:nlevel
        Alev = view(Adata, :, :, level)
        nofflev = (level - 1) * nsubrow
        clev = view(c, (nofflev + 1):(nofflev + nsubrow))

        # First term: cpi · (AΛ)pj · c
        fill!(Aλrow, zero(T))

        for subcol in 1:nsubcol
            for k in 1:nsubcol
                Aλrow[subcol] += Alev[pj, k] * λ[k, subcol]
            end
        end

        out += clev[pi] * dot(Aλrow, clev)

        # Second term: c · (Λ'A)pi · cpj
        fill!(λAcol, zero(T))

        for subrow in 1:nsubrow
            for k in 1:nsubrow
                λAcol[subrow] += λ[k, subrow] * Alev[k, pi]
            end
        end

        out += dot(λAcol, clev) * clev[pj]
    end

    return out
end

# Quadratic form for off-diagonal block: cr' ∂Ωrk ck
function _quadform_offdiag_block(
    crow::AbstractVector{T},
    ccol::AbstractVector{T},
    A::AbstractMatrix{T},
    Λrow::ReMat{T, 1},
    Λcol::ReMat{T, 1},
    pi::Int,
    pj::Int,
    iscolpar::Bool,
) where {T}
    #
    # Ωrk = λr A λk   →   ∂Ωrk/∂λk = λr A   or   ∂Ωrk/∂λr = A λk
    #
    # cr'(∂Ω)ck = Σij cri · scale · Aij · ckj
    #
    λrow = only(Λrow.λ.data)
    λcol = only(Λcol.λ.data)
    scale = iscolpar ? λrow : λcol
    out = zero(T)

    if A isa Diagonal
        for (cri, Aii, cki) in zip(crow, A.diag, ccol)
            out += cri * scale * Aii * cki
        end
    elseif A isa SparseMatrixCSC
        for col in 1:size(A, 2)
            for p in nzrange(A, col)
                row = rowvals(A)[p]
                Aij = nonzeros(A)[p]
                out += crow[row] * scale * Aij * ccol[col]
            end
        end
    elseif A isa BlockedSparse
        Acsc = A.cscmat
        for col in 1:size(Acsc, 2)
            for p in nzrange(Acsc, col)
                row = rowvals(Acsc)[p]
                Aij = nonzeros(Acsc)[p]
                out += crow[row] * scale * Aij * ccol[col]
            end
        end
    else
        for col in axes(A, 2), row in axes(A, 1)
            out += crow[row] * scale * A[row, col] * ccol[col]
        end
    end

    return out
end

function _quadform_offdiag_block(
    crow::AbstractVector{T},
    ccol::AbstractVector{T},
    A::BlockedSparse{T},
    Λrow::ReMat{T, Srow},
    Λcol::ReMat{T, Scol},
    pi::Int,
    pj::Int,
    iscolpar::Bool,
) where {T, Srow, Scol}
    #
    # Ωrk = Λr' A Λk   →   ∂Ω/∂Λkij = Λr' A Eij   or   ∂Ω/∂Λrij = Eji A Λk
    #
    λrow = Λrow.λ
    λcol = Λcol.λ
    Acsc = A.cscmat
    out = zero(T)

    nsubrow = size(λrow, 1)
    nsubcol = size(λcol, 1)

    for loccol in 1:size(Acsc, 2)
        for ptr in nzrange(Acsc, loccol)
            locrow = rowvals(Acsc)[ptr]
            Aij = nonzeros(Acsc)[ptr]

            rowlevel = (locrow - 1) ÷ nsubrow + 1
            rowsub = (locrow - 1) % nsubrow + 1
            collevel = (loccol - 1) ÷ nsubcol + 1
            colsub = (loccol - 1) % nsubcol + 1

            if iscolpar
                if colsub == pj
                    for k in 1:nsubrow
                        out += crow[locrow] * λrow[k, rowsub] * Aij * ccol[(collevel - 1) * nsubcol + pi]
                    end
                end
            else
                if rowsub == pj
                    for k in 1:nsubcol
                        out += crow[(rowlevel - 1) * nsubrow + pi] * Aij * λcol[k, colsub] * ccol[loccol]
                    end
                end
            end
        end
    end

    return out
end

function _quadform_fe_cross(
    ck::AbstractVector{T},
    cFE::AbstractVector{T},
    AFEk::Matrix{T},  # X'Zk, size (nFE × nk)
    Λk::ReMat{T, 1},
    pi::Int,
    pj::Int,
) where {T}
    #
    # Scalar RE:  ∂ΩZFk/∂λ = Zk'X = AFEk'   →   ck' AFEk' cFE = (AFEk ck)' cFE
    #
    out = zero(T)

    for (row, cFErow) in enumerate(cFE)
        Acrow = zero(T)

        for (col, ckcol) in enumerate(ck)
            Acrow += AFEk[row, col] * ckcol
        end

        out += Acrow * cFErow
    end

    return out
end

function _quadform_fe_cross(
    ck::AbstractVector{T},
    cFE::AbstractVector{T},
    AFEk::Matrix{T},  # X'Zk, size (nFE × nk)
    Λk::ReMat{T, S},
    pi::Int,
    pj::Int,
) where {T, S}
    #
    # Vector RE:  ∂Λk'/∂θij = Epj,pi   →   ∂ΩZFk = Epj,pi Zk'X
    #
    # ck' Epj,pi Zk'X cFE = Σlev ck_lev,pj · AFEk[:,lev,pi]' · cFE
    #
    nlevel = nlevs(Λk)
    out = zero(T)

    for level in 1:nlevel
        cklevpj = ck[(level - 1) * S + pj]
        loccol = (level - 1) * S + pi

        for (row, cFErow) in enumerate(cFE)
            out += cklevpj * AFEk[row, loccol] * cFErow
        end
    end

    return out
end

function _solve_Lt(F::ChordalCholesky{UPLO, T}) where {UPLO, T}
    #
    # c = P' Linv' elast = P \ (L' \ elast)
    #
    n = size(F.L, 1)
    elast = zeros(T, n)
    elast[end] = one(T)
    return F.P \ (F.L' \ elast)
end

function _resid_gradient(
    M::LinearMixedModel{T},
    F::ChordalCholesky{UPLO, T},
    W::ChordalWorkspace{T},
    c::Vector{T},
) where {UPLO, T}
    #
    # Residual gradient:  dof · c' dΩ/dθp c
    #
    # where c = Linv' elast, and  c'dΩc = cZZ'dΩZZ cZZ + 2·cZZ'dΩZF cFE
    #
    dof = ssqdenom(M)
    q = _nzz(M, F)
    cZZ = view(c, 1:q)
    cFE = view(c, (q + 1):length(c))

    g = zeros(T, length(M.parmap))

    for p in eachindex(g)
        quadform = _quadform_Ωdot(cZZ, cFE, M, W, p)
        g[p] = dof * quadform
    end

    return g
end

"""
    _objective_gradient(M::LinearMixedModel{T}, F::ChordalCholesky, W::ChordalWorkspace{T})

Compute the objective and its gradient (ML or REML).

Returns (objective, gradient) where objective = log|Ω_ZZ| + dof*log(pwrss).
"""
function _objective_gradient(
    M::LinearMixedModel{T},
    F::ChordalCholesky{UPLO, T},
    W::ChordalWorkspace{T},
) where {UPLO, T}
    #
    # Workflow:
    #   1. Ω = Λ'AΛ + I  (transform)
    #   2. L L' = Ω      (cholesky)
    #   3. c = Linv' elast, pwrss = Llast²
    #   4. selinv → ΩZZinv
    #   5. g = tr(ΩZZinv dΩZZ) + dof·c'dΩc
    #
    dof = ssqdenom(M)

    # Transform and factorize
    _transform!(M, F, W)
    cholesky!(F)

    # Extract log|ΩZZ| and pwrss
    logdetZZ = _logdet(M, F)
    pwrss = _pwrss(F)

    # c = P \ (L' \ elast) before selinv
    c = _solve_Lt(F)

    # Prepare and run selinv
    q = _nzz(M, F)
    _prepare_selinv_ZZ!(F, q)
    selinv!(F)

    # Gradient = logdet part + residual part
    g = _logdet_gradient(M, F, W) .+ _resid_gradient(M, F, W, c)

    # Objective
    objective = logdetZZ + dof * log(pwrss)

    return objective, g
end
