LinAlg.Ac_ldiv_B!{T}(D::Diagonal{T}, B) = A_ldiv_B!(D, B)

function LinAlg.A_ldiv_B!{T}(D::Diagonal{T}, B::Diagonal{T})
    if size(D) ≠ size(B)
        throw(DimensionMismatch("size(D) ≠ size(B)"))
    end
    map!(/, B.diag, B.diag, D.diag)
    B
end

function LinAlg.A_ldiv_B!{T}(D::Diagonal{T}, B::SparseMatrixCSC{T})
    dd = D.diag
    if length(dd) ≠ size(B, 1)
        throw(DimensionMismatch("size(D,2) ≠ size(B,1)"))
    end
    vals = nonzeros(B)
    rows = rowvals(B)
    @inbounds for k in eachindex(vals)
        vals[k] /= dd[rows[k]]
    end
    B
end

function rowlengths(L::LowerTriangular)
    ld = L.data
    [(sl = view(ld, i, 1:i); sqrt(dot(sl, sl))) for i in 1:size(L, 1)]
end
