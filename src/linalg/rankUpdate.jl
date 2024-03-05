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
        "We haven't implemented a method for $(typeof(C)), $(typeof(a)). Please file an issue on GitHub.",
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

function rankUpdate!(
    C::HermOrSym{T,UniformBlockDiagonal{T}}, A::StridedMatrix{T}, α, β
) where {T}
    Cdat = C.data.data
    require_one_based_indexing(Cdat, A)
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
                AtAij = muladd(A[iind, idx], A[jind, idx], AtAij)
            end
            Cdat[i, j, k] = muladd(α, AtAij, Cdat[i, j, k])
        end
    end

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

function rankUpdate!(
    C::HermOrSym{T,UniformBlockDiagonal{T}}, A::BlockedSparse{T,S}, α, β
) where {T,S}
    Ac = A.cscmat
    cp = Ac.colptr
    all(==(S), diff(cp)) ||
        throw(ArgumentError("Columns of A must have exactly $S nonzeros"))
    Cdat = C.data.data
    require_one_based_indexing(Ac, Cdat)

    j, k, l = size(Cdat)
    S == j == k && div(Ac.m, S) == l ||
        throw(DimensionMismatch("div(A.cscmat.m, S) ≠ size(C.data.data, 3)"))
    nz = Ac.nzval
    rv = Ac.rowval

    @inbounds for j in axes(Ac, 2)
        nzr = nzrange(Ac, j)
        BLAS.syr!('L', α, view(nz, nzr), view(Cdat, :, :, div(rv[last(nzr)], S)))
    end

    return C
end
