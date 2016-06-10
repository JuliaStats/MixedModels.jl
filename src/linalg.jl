LinAlg.Ac_ldiv_B!{T}(D::Diagonal{T}, B) = A_ldiv_B!(D, B)

function LinAlg.A_ldiv_B!{T}(D::Diagonal{T}, B::Diagonal{T})
    dd = D.diag
    bd = B.diag
    if length(dd) ≠ length(bd)
        throw(DimensionMismatch("size(D,2) ≠ size(B,1)"))
    end
    for j in eachindex(bd)
        bd[j] /= dd[j]
    end
    B
end

function LinAlg.A_ldiv_B!{T}(D::Diagonal{T}, B::SparseMatrixCSC{T})
    m, n = size(B)
    dd = D.diag
    if length(dd) ≠ m
        throw(DimensionMismatch("size(D,2) ≠ size(B,1)"))
    end
    nzv = nonzeros(B)
    rv = rowvals(B)
    for j in 1:n, k in nzrange(B,j)
        nzv[k] /= dd[rv[k]]
    end
    B
end

function LinAlg.Ac_ldiv_B!{T}(A::UpperTriangular{T,HBlkDiag{T}}, B::DenseMatrix{T})
    m, n = size(B)
    aa = A.data.arr
    r, s, t = size(aa)
    if m ≠ Compat.LinAlg.checksquare(A)
        throw(DimensionMismatch("size(A,2) ≠ size(B,1)"))
    end
    scv = Array(T, (r,))
    scm = Array(T, (r, r))
    for k in 1:t
        for j in 1 : r, i in 1 : j
            scm[i, j] = aa[i, j, k]
        end
        rowoff = (k - 1) * r
        for j in 1:n
            for i in 1 : r
                scv[i] = B[rowoff + i, j]
            end
            BLAS.trsv!('U', 'T', 'N', scm, scv)
            for i in 1 : r
                B[rowoff + i, j] = scv[i]
            end
        end
    end
    B
end

function LinAlg.Ac_ldiv_B!{T}(A::UpperTriangular{T,HBlkDiag{T}}, B::SparseMatrixCSC{T})
    m, n = size(B)
    aa = A.data.arr
    r, s, t = size(aa)
    if r ≠ s || r * t ≠ m
        throw(DimensionMismatch("size(A,2) ≠ size(B,1)"))
    end
    rows = rowvals(B)
    vals = nonzeros(B)
    for j in 1 : n
        nzrj = nzrange(B, j)
        q, r = divrem(length(nzrj), s)
        if r ≠ 0
            error("length(nzrange(B, $j)) is not divisible by $s")
        end
        for b in 1 : q
            subrng = nzrj[(b - 1) * s + (1 : s)]
            rr = sub(rows, subrng)
            if any(d -> d ≠ 1, diff(rr))
                error("rows of block $b in column $j are not contiguous")
            end
            q1, r1 = divrem(rr[end], s)
            if r1 ≠ 0
                error("rows of block $b in column $j do not end in a multiple of $s")
            end
            BLAS.trsv!('U', 'T', 'N', sub(aa, :, :, q1), sub(vals, subrng))
        end
    end
    B
end

function rowlengths(L::LowerTriangular)
    ld = L.data
    [(sl = slice(ld, i, 1:i); sqrt(dot(sl, sl))) for i in 1:size(L, 1)]
end
