"""
    rankUpdate!(C, A)
    rankUpdate!(C, A, α)
    rankUpdate!(C, A, α, β)

A rank-k update, C := α*A'A + β*C, of a Hermitian (Symmetric) matrix.

`α` and `β` both default to 1.0.  When `α` is -1.0 this is a downdate operation.
The name `rankUpdate!` is borrowed from [https://github.com/andreasnoack/LinearAlgebra.jl]
"""
function rankUpdate! end

function rankUpdate!(C::AbstractMatrix, a::AbstractArray, α, β)
    return error(
        "We haven't implemented a method for $(typeof(C)), $(typeof(a)). Please file an issue on GitHub."
    )
end

function MixedModels.rankUpdate!(
    C::Hermitian{T,Diagonal{T,Vector{T}}}, A::Diagonal{T,Vector{T}}, α, β
) where {T}
    Cdiag = C.data.diag
    Adiag = A.diag
    @inbounds for idx in eachindex(Cdiag, Adiag)
        Cdiag[idx] = muladd(β, Cdiag[idx], α * abs2(Adiag[idx]))
    end
    return C
end

function rankUpdate!(C::HermOrSym{T,S}, a::StridedVector{T}, α, β) where {T,S}
    Cd = C.data
    isone(β) || rmul!(C.uplo == 'L' ? LowerTriangular(Cd) : UpperTriangular(Cd), β)
    BLAS.syr!(C.uplo, T(α), a, Cd)
    return C  ## to ensure that the return value is HermOrSym
end

function rankUpdate!(C::HermOrSym{T,S}, A::StridedMatrix{T}, α, β) where {T,S}
    BLAS.syrk!(C.uplo, 'N', T(α), A, T(β), C.data)
    return C
end

"""
    _columndot(rv, nz, rngi, rngj)

Return the dot product of two columns, with `nzrange`s `rngi` and `rngj`, of a sparse matrix defined by rowvals `rv` and nonzeros `nz`
"""
function _columndot(rv, nz, rngi, rngj)
    accum = zero(eltype(nz))
    (isempty(rngi) || isempty(rngj)) && return accum
    ni, nj = length(rngi), length(rngj)
    i = j = 1
    while i ≤ ni && j ≤ nj
        @inbounds ri, rj = rv[rngi[i]], rv[rngj[j]]
        if ri == rj
            @inbounds accum = muladd(nz[rngi[i]], nz[rngj[j]], accum)
            i += 1
            j += 1
        elseif ri < rj
            i += 1
        else
            j += 1
        end
    end
    return accum
end

function rankUpdate!(C::HermOrSym{T,S}, A::SparseMatrixCSC{T}, α, β) where {T,S}
    require_one_based_indexing(C, A)
    m, n = size(A)
    Cd, rv, nz = C.data, A.rowval, A.nzval
    lower = C.uplo == 'L'
    (lower ? m : n) == size(C, 2) || throw(DimensionMismatch())
    isone(β) || rmul!(lower ? LowerTriangular(Cd) : UpperTriangular(Cd), β)
    if lower
        @inbounds for jj in axes(A, 2)
            rangejj = nzrange(A, jj)
            lenrngjj = length(rangejj)
            for (k, j) in enumerate(rangejj)
                anzj = α * nz[j]
                rvj = rv[j]
                for i in k:lenrngjj
                    kk = rangejj[i]
                    Cd[rv[kk], rvj] = muladd(nz[kk], anzj, Cd[rv[kk], rvj])
                end
            end
        end
    else
        @inbounds for j in axes(C, 2)
            rngj = nzrange(A, j)
            for i in 1:(j - 1)
                Cd[i, j] = muladd(α, _columndot(rv, nz, nzrange(A, i), rngj), Cd[i, j])
            end
            Cd[j, j] = muladd(α, sum(i -> abs2(nz[i]), rngj), Cd[j, j])
        end
    end
    return C
end

function rankUpdate!(C::HermOrSym, A::BlockedSparse, α, β)
    return rankUpdate!(C, sparse(A), α, β)
end

function rankUpdate!(
    C::HermOrSym{T,Diagonal{T,Vector{T}}}, A::StridedMatrix{T}, α, β
) where {T}
    Cdiag = C.data.diag
    require_one_based_indexing(Cdiag, A)
    length(Cdiag) == size(A, 1) || throw(DimensionMismatch())
    isone(β) || rmul!(Cdiag, β)

    @inbounds for i in eachindex(Cdiag)
        Cdiag[i] = muladd(α, sum(abs2, view(A, i, :)), Cdiag[i])
    end

    return C
end

# Faces at or above this size use the BLAS-3 syrk! path; below it the per-face BLAS
# call overhead outweighs the arithmetic savings and the scalar loop is faster.
# (Empirically the crossover sits between blksize 3 and 4; see the benchmark notes.)
const _UBD_BLAS3_MIN = 4

# scalar per-face accumulation Cₖ := α Aₖ Aₖᵀ + β Cₖ over the lower triangle;
# used for eltypes without BLAS support and for small faces
function _ubd_rankupdate!(Cdat, A, α, β)
    isone(β) || rmul!(Cdat, β)
    blksize = size(Cdat, 1)
    for k in axes(Cdat, 3)
        offset = (k - 1) * blksize
        for i in axes(Cdat, 1), j in 1:i
            iind = offset + i
            jind = offset + j
            AtAij = zero(eltype(Cdat))
            for idx in axes(A, 2)
                # because the second multiplicant is from A', swap index order
                AtAij = muladd(A[iind, idx], A[jind, idx], AtAij)
            end
            Cdat[i, j, k] = muladd(α, AtAij, Cdat[i, j, k])
        end
    end
    return Cdat
end

# Each S×S face of the UniformBlockDiagonal owns a contiguous row band of the dense
# A, so the whole face update Cₖ := α Aₖ Aₖᵀ + β Cₖ is a single BLAS-3 syrk! on a
# zero-copy view.  syrk! scales and writes only the C.uplo triangle; the opposite
# triangle keeps its stale value, which is harmless because the block is always
# consumed through the Hermitian/Symmetric wrapper.
function rankUpdate!(
    C::HermOrSym{T,UniformBlockDiagonal{T}}, A::StridedMatrix{T}, α, β
) where {T<:BlasFloat}
    Cdat = C.data.data
    require_one_based_indexing(Cdat, A)
    blksize = size(Cdat, 1)
    size(A, 1) == blksize * size(Cdat, 3) ||
        throw(DimensionMismatch("size(A, 1) ≠ blksize * nblocks"))

    if blksize < _UBD_BLAS3_MIN
        _ubd_rankupdate!(Cdat, A, α, β)
        return C
    end

    for k in axes(Cdat, 3)
        rows = ((k - 1) * blksize + 1):(k * blksize)
        BLAS.syrk!(C.uplo, 'N', T(α), view(A, rows, :), T(β), view(Cdat, :, :, k))
    end

    return C
end

# generic fallback for eltypes without BLAS support
function rankUpdate!(
    C::HermOrSym{T,UniformBlockDiagonal{T}}, A::StridedMatrix{T}, α, β
) where {T}
    Cdat = C.data.data
    require_one_based_indexing(Cdat, A)
    _ubd_rankupdate!(Cdat, A, α, β)
    return C
end

function rankUpdate!(
    C::HermOrSym{T,Diagonal{T,Vector{T}}}, A::SparseMatrixCSC{T}, α, β
) where {T}
    dd = C.data.diag
    require_one_based_indexing(dd, A)
    A.m == length(dd) || throw(DimensionMismatch())
    isone(β) || rmul!(dd, β)
    all(isone.(diff(A.colptr))) ||
        throw(ArgumentError("Columns of A must have exactly 1 nonzero"))

    for (r, nz) in zip(rowvals(A), nonzeros(A))
        dd[r] = muladd(α, abs2(nz), dd[r])
    end

    return C
end

function rankUpdate!(C::HermOrSym{T,Diagonal{T}}, A::BlockedSparse{T}, α, β) where {T}
    return rankUpdate!(C, sparse(A), α, β)
end

# Each column of A has exactly S nonzeros, all landing in a single S×S face of C, so
# each face update is Cₖ := α Aₖ Aₖᵀ + β Cₖ where Aₖ collects that face's columns.
# With exactly S nonzeros per column the nonzeros form a dense S×ncols panel
# (reshape shares the nzval buffer, no copy), and columns of a face are contiguous
# in practice (a column touching one face is a nested relationship), so a single
# BLAS-3 syrk! per maximal contiguous run of equal face replaces the many BLAS-2
# rank-1 syr! calls — several-fold faster and allocation-free.  Non-contiguous
# orderings simply split a face across runs (still correct, never worse than the
# per-column path), since β is applied once up front and every syrk! accumulates.
function rankUpdate!(
    C::HermOrSym{T,UniformBlockDiagonal{T}}, A::BlockedSparse{T,S}, α, β
) where {T,S}
    Ac = A.cscmat
    cp = Ac.colptr
    ncols = size(Ac, 2)
    all(j -> cp[j + 1] - cp[j] == S, axes(Ac, 2)) ||  # allocation-free vs diff(cp)
        throw(ArgumentError("Columns of A must have exactly $S nonzeros"))
    Cdat = C.data.data
    require_one_based_indexing(Ac, Cdat)

    r, c, l = size(Cdat)
    S == r == c && div(Ac.m, S) == l ||
        throw(DimensionMismatch("div(A.cscmat.m, S) ≠ size(C.data.data, 3)"))
    rv = Ac.rowval
    panel = reshape(Ac.nzval, S, ncols)  # column j ↔ face div(rv[j*S], S)

    isone(β) || rmul!(Cdat, β)
    ncols == 0 && return C

    runstart = 1
    curface = div(rv[S], S)
    @inbounds for j in 2:ncols
        face = div(rv[j * S], S)
        if face != curface
            BLAS.syrk!('L', 'N', T(α), view(panel, :, runstart:(j - 1)),
                one(T), view(Cdat, :, :, curface))
            runstart = j
            curface = face
        end
    end
    @inbounds BLAS.syrk!('L', 'N', T(α), view(panel, :, runstart:ncols),
        one(T), view(Cdat, :, :, curface))

    return C
end
