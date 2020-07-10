"""
    rankUpdate!(C, A)
    rankUpdate!(C, A, α)
    rankUpdate!(C, A, α, β)

A rank-k update, C := β*C + α*A'A, of a Hermitian (Symmetric) matrix.

`α` and `β` both default to 1.0.  When `α` is -1.0 this is a downdate operation.
The name `rankUpdate!` is borrowed from [https://github.com/andreasnoack/LinearAlgebra.jl]
The order of the arguments
"""
function rankUpdate! end

function rankUpdate!(
    C::AbstractMatrix{T},
    A::AbstractMatrix{T},
    α=true,
    β=true) where {T}

    m, n = size(A)
    m == size(C, 2) || throw(DimensionMismatch(""))
    @info "it is surprising that this function is called - please report a use case as a MixedModels issue"
    @warn "using generic method, this will be slower than usual"

    A = A'A

    for (i, el) in enumerate(C)
        C[i] = β * el + α * A[i]
    end
    C
end

function rankUpdate!(
    C::HermOrSym{T,S},
    a::StridedVector{T},
    α = true,
) where {T<:BlasReal,S<:StridedMatrix}
    BLAS.syr!(C.uplo, T(α), a, C.data)
    C  ## to ensure that the return value is HermOrSym
end

function rankUpdate!(
    C::HermOrSym{T,S},
    A::StridedMatrix{T},
    α = true,
    β = true,
) where {T<:BlasReal,S<:StridedMatrix}
    BLAS.syrk!(C.uplo, 'N', T(α), A, T(β), C.data)
    C
end

function rankUpdate!(
    C::HermOrSym{T,Matrix{T}},
    A::SparseMatrixCSC{T},
    α = true,
    β = true,
) where {T}
    m, n = size(A)
    m == size(C, 2) || throw(DimensionMismatch(""))
    C.uplo == 'L' || throw(ArgumentError("C.uplo must be 'L'"))
    Cd = C.data
    isone(β) || rmul!(LowerTriangular(Cd), β)
    rv = rowvals(A)
    nz = nonzeros(A)
    @inbounds for jj = 1:n
        rangejj = nzrange(A, jj)
        lenrngjj = length(rangejj)
        for (k, j) in enumerate(rangejj)
            anzj = α * nz[j]
            rvj = rv[j]
            for i = k:lenrngjj
                kk = rangejj[i]
                Cd[rv[kk], rvj] += nz[kk] * anzj
            end
        end
    end
    C
end

rankUpdate!(C::HermOrSym, A::BlockedSparse, α = true, β = true) =
    rankUpdate!(C, sparse(A), α, β)

function rankUpdate!(
    C::Diagonal{T},
    A::SparseMatrixCSC{T},
    α = true,
    β = true,
) where {T<:Number}
    m, n = size(A)
    dd = C.diag
    length(dd) == m || throw(DimensionMismatch(""))
    isone(β) || rmul!(dd, β)
    nz = nonzeros(A)
    rv = rowvals(A)
    @inbounds for j = 1:n
        nzr = nzrange(A, j)
        if !isempty(nzr)
            length(nzr) == 1 || throw(ArgumentError("A*A' has off-diagonal elements"))
            k = nzr[1]
            dd[rv[k]] += α * abs2(nz[k])
        end
    end
    C
end

rankUpdate!(C::Diagonal{T}, A::BlockedSparse{T}, α = true, β = true) where {T<:Number} =
    rankUpdate!(C, sparse(A), α, β)

function rankUpdate!(
    C::HermOrSym{T,UniformBlockDiagonal{T}},
    A::BlockedSparse{T,S},
    α = true,
) where {T,S}
    Ac = A.cscmat
    cp = Ac.colptr
    all(diff(cp) .== S) || throw(ArgumentError("Each column of A must contain exactly S nonzeros"))
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

#=
rankUpdate!(C::HermOrSym{T,Matrix{T}}, A::BlockedSparse{T}, α = true) where {T} =
    rankUpdate!(C, A.cscmat, α)
=#

function rankUpdate!(
    C::Diagonal{T,S},
    A::Diagonal{T,S},
    α::Number = true,
    β::Number = true,
) where {T,S}
    length(C.diag) == length(A.diag) || throw(DimensionMismatch("length(C.diag) ≠ length(A.diag)"))

    C.diag .= β .* C.diag .+ α .* abs2.(A.diag)
    C
end


function rankUpdate!(
    C::HermOrSym{T,Matrix{T}},
    A::Diagonal{T},
    α::Number = true,
    β::Number = true,
) where {T}
    m, n = size(A)
    m == size(C, 2) || throw(DimensionMismatch(""))
    adiag = A.diag
    cdata = C.data
    for (i, a) in zip(diagind(cdata), adiag)
        cdata[i] = β * cdata[i] + α * abs2(a)
    end
    C
end
