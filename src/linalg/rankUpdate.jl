"""
    rankUpdate!(C, A)
    rankUpdate!(C, A, α)
    rankUpdate!(C, A, α, β)

A rank-k update, C := β*C + α*A'A, of a Hermitian (Symmetric) matrix.

`α` and `β` both default to 1.0.  When `α` is -1.0 this is a downdate operation.
The name `rankUpdate!` is borrowed from [https://github.com/andreasnoack/LinearAlgebra.jl]
"""
function rankUpdate! end

function rankUpdate!(C::AbstractMatrix, a::AbstractArray, α, β)
    error("We haven't implemented a method for $(typeof(C)), $(typeof(a)). Please file an issue on GitHub.")
end

function rankUpdate!(C::HermOrSym{T,S}, a::StridedVector{T}, α, β) where {T,S}
    isone(β) || throw(ArgumentError("isone(β) is false"))
    BLAS.syr!(C.uplo, T(α), a, C.data)
    C  ## to ensure that the return value is HermOrSym
end

function rankUpdate!(C::HermOrSym{T,S}, A::StridedMatrix{T}, α, β) where {T,S}
    BLAS.syrk!(C.uplo, 'N', T(α), A, T(β), C.data)
    C
end

function rankUpdate!(C::HermOrSym{T,S}, A::SparseMatrixCSC{T}, α, β) where {T,S}
    A.m == size(C, 2) || throw(DimensionMismatch())
    C.uplo == 'L' || throw(ArgumentError("C.uplo must be 'L'"))
    Cd, rv, nz = C.data, rowvals(A), nonzeros(A)
    isone(β) || rmul!(LowerTriangular(Cd), β)
    @inbounds for jj in 1:A.n
        rangejj = nzrange(A, jj)
        lenrngjj = length(rangejj)
        for (k, j) in enumerate(rangejj)
            anzj = α * nz[j]
            colj = view(Cd, :, rv[j])
            for i in k:lenrngjj
                kk = rangejj[i]
                colj[rv[kk]] += nz[kk] * anzj
            end
        end
    end
    C
end

function rankUpdate!(C::HermOrSym, A::BlockedSparse, α, β)
    rankUpdate!(C, sparse(A), α, β)
end

function rankUpdate!(C::HermOrSym{T,Diagonal{T,Vector{T}}}, A::StridedMatrix{T}, α, β) where {T,S}
    Cdiag = C.data.diag
    @. Cdiag = β * Cdiag

    for i in 1:length(Cdiag)
        Arow = view(A, i, :)
        Cdiag[i] = Cdiag[i] + α * Arow'Arow
    end

    C
end

function rankUpdate!(C::HermOrSym{T,UniformBlockDiagonal{T}}, A::StridedMatrix{T}, α, β) where {T,S}
    Cdat = C.data.data
    isone(β) || (Cdat .*= β)
    blksize = size(Cdat, 1)

    for k in axes(Cdat, 3)
        ioffset = (k - 1) * blksize
        joffset = (k - 1) * blksize
        for i in 1:blksize, j in 1:i
            iind = ioffset + i
            jind = joffset + j
            Arow = view(A, iind, :)
            Acol = view(A, jind, :) # because the column comes from A'
            Cdat[i,j,k] = Cdat[i,j,k] + α * Arow'Acol
        end
    end

    C
end

function rankUpdate!(
    C::HermOrSym{T,Diagonal{T,Vector{T}}},
    A::SparseMatrixCSC{T},
    α,
    β,
) where {T}
    dd = C.data.diag
    A.m == length(dd) || throw(DimensionMismatch())
    isone(β) || rmul!(dd, β)
    all(isone.(diff(A.colptr))) || throw(ArgumentError("Columns of A must have exactly 1 nonzero"))
    for (r, nz) in zip(rowvals(A), nonzeros(A))
        dd[r] += α * abs2(nz)
    end
    C
end

function rankUpdate!(C::HermOrSym{T,Diagonal{T}}, A::BlockedSparse{T}, α, β) where {T}
    rankUpdate!(C, sparse(A), α, β)
end

function rankUpdate!(
    C::HermOrSym{T,UniformBlockDiagonal{T}},
    A::BlockedSparse{T,S},
    α,
    β,
) where {T,S}
    Ac = A.cscmat
    cp = Ac.colptr
    all(diff(cp) .== S) || throw(ArgumentError("Columns of A must have exactly $S nonzeros"))
    Cdat = C.data.data
    j, k, l = size(Cdat)
    S == j == k && div(Ac.m, S) == l ||
    throw(DimensionMismatch("div(A.cscmat.m, S) ≠ size(C.data.data, 3)"))
    nz = Ac.nzval
    rv = Ac.rowval
    for j = 1:Ac.n
        nzr = nzrange(Ac, j)
        BLAS.syr!('L', α, view(nz, nzr), view(Cdat, :, :, div(rv[last(nzr)], S)))
    end
    C
end
#=  I don't think Diagonal A can occur after the terms with the same grouping factor have been amalgamated.
function rankUpdate!(C::HermOrSym{T,Diagonal{T}}, A::Diagonal{T}, α, β) where {T}
    Cdiag = C.data.diag
    if length(Cdiag) ≠ length(A.diag)
        throw(DimensionMismatch("length(C.data.diag) ≠ length(A.diag)"))
    end

    Cdiag .= β .* Cdiag .+ α .* abs2.(A.diag)
    C
end

function rankUpdate!(C::HermOrSym{T,Matrix{T}}, A::Diagonal{T}, α, β) where {T}
    Adiag, Cdata = A.diag, C.data
    length(Adiag) == size(C, 2) || throw(DimensionMismatch())
    for (i, a) in zip(diagind(Cdata), Adiag)
        Cdata[i] = β * Cdata[i] + α * abs2(a)
    end
    C
end
=#
