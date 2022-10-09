function LinearAlgebra.mul!(
    C::Matrix{T},
    blkA::BlockedSparse{T},
    adjB::Adjoint{T,<:BlockedSparse{T}},
    α::Number,
    β::Number,
) where {T}
    A = blkA.cscmat
    B = adjB.parent.cscmat
    B.m == size(C, 2) && A.m == size(C, 1) && A.n == B.n || throw(DimensionMismatch(""))
    anz = nonzeros(A)
    arv = rowvals(A)
    bnz = nonzeros(B)
    brv = rowvals(B)
    isone(β) || rmul!(C, β)
    @inbounds for j in 1:(A.n)
        for ib in nzrange(B, j)
            αbnz = α * bnz[ib]
            jj = brv[ib]
            for ia in nzrange(A, j)
                C[arv[ia], jj] += anz[ia] * αbnz
            end
        end
    end
    return C
end

function LinearAlgebra.mul!(
    C::StridedVecOrMat{T},
    A::StridedVecOrMat{T},
    adjB::Adjoint{T,<:BlockedSparse{T}},
    α::Number,
    β::Number,
) where {T}
    return mul!(C, A, adjoint(adjB.parent.cscmat), α, β)
end

function LinearAlgebra.mul!(
    C::StridedVector{T},
    adjA::Adjoint{T,<:BlockedSparse{T}},
    B::StridedVector{T},
    α::Number,
    β::Number,
) where {T}
    return mul!(C, adjoint(adjA.parent.cscmat), B, α, β)
end

function LinearAlgebra.ldiv!(
    A::UpperTriangular{T,<:Adjoint{T,UniformBlockDiagonal{T}}}, B::StridedVector{T}
) where {T}
    adjA = A.data
    length(B) == size(A, 2) || throw(DimensionMismatch(""))
    Adat = adjA.parent.data
    m, n, k = size(Adat)
    bb = reshape(B, (n, k))
    for j in axes(Adat, 3)
        ldiv!(UpperTriangular(adjoint(view(Adat, :, :, j))), view(bb, :, j))
    end
    return B
end

function LinearAlgebra.rdiv!(
    A::Matrix{T}, B::UpperTriangular{T,<:Adjoint{T,UniformBlockDiagonal{T}}}
) where {T}
    m, n = size(A)
    Bd = B.data.parent
    Bdd = Bd.data
    r, s, blk = size(Bdd)
    n == size(Bd, 1) && r == s || throw(DimensionMismatch())
    for b in axes(Bd.data, 3)
        coloffset = (b - 1) * s
        rdiv!(
            view(A, :, (coloffset + 1):(coloffset + s)),
            UpperTriangular(adjoint(view(Bdd, :, :, b))),
        )
    end
    return A
end

function LinearAlgebra.rdiv!(
    A::BlockedSparse{T,S,P}, B::UpperTriangular{T,<:Adjoint{T,UniformBlockDiagonal{T}}}
) where {T,S,P}
    Bpd = B.data.parent
    Bdat = Bpd.data
    j, k, l = size(Bdat)
    cbpt = A.colblkptr
    nzv = A.cscmat.nzval
    P == j == k && length(cbpt) == (l + 1) || throw(DimensionMismatch(""))
    for j in axes(Bdat, 3)
        rdiv!(
            reshape(view(nzv, cbpt[j]:(cbpt[j + 1] - 1)), :, P),
            UpperTriangular(adjoint(view(Bdat, :, :, j))),
        )
    end
    return A
end

function RectangularFullPacked.TriangularRFP(d::Diagonal{T}, uplo::Symbol=:U; transr::Symbol=:N) where {T}
    ul = first(string(uplo))
    if ul ∉ "UL"
        throw(ArgumentError("uplo = $uplo should be :U or :L"))
    end
    tr = first(string(transr))
    if tr ∉ (T <: Complex ? "NC" : "NT")
        throw(ArgumentError("transr = $transr should be :N or :(T <: Complex ? :C : :T)"))
    end
    ddiag = d.diag
    rfdims = RectangularFullPacked._parentsize(length(ddiag), tr ≠ 'N')
    val = TriangularRFP(zeros(T, first(rfdims), last(rfdims)), tr, ul)
    for (i, d) in enumerate(ddiag)
        val[i, i] = d
    end
    return val
end

function RectangularFullPacked.TriangularRFP(A::UniformBlockDiagonal{T}, uplo::Symbol=:U; transr::Symbol=:N) where {T}
    ul = uppercase(first(string(uplo)))
    if ul ≠ 'L'
        throw(ArgumentError("uplo = $uplo should be :L"))
    end
    tr = first(string(transr))
    if tr ∉ (T <: Complex ? "NC" : "NT")
        throw(ArgumentError("transr = $transr should be :N or :(T <: Complex ? :C : :T)"))
    end
    Ad = A.data
    n = LinearAlgebra.checksquare(A)
    neven = iseven(n)
    k, l = RectangularFullPacked._parentsize(n, tr ≠ 'N')
    Cd = zeros(T, k, l)
    val = TriangularRFP(Cd, tr, ul)
    r = size(Ad, 1)
    for kk in axes(Ad, 3)
        offset = (kk - 1) * r 
        for j in axes(Ad, 2)
            for i in j:r
                Cd[_packedinds(i+offset, j+offset, neven, l)...] = Ad[i, j, kk]
            end
        end
    end
    return val
end

@static if VERSION < v"1.7.0-DEV.1188" # julialang sha e0ecc557a24eb3338b8dc672d02c98e8b31111fa
    pivoted_qr(A; kwargs...) = qr(A, Val(true); kwargs...)
else
    pivoted_qr(A; kwargs...) = qr(A, ColumnNorm(); kwargs...)
end
