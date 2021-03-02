"""
    statsrank(x::Matrix{T}, ranktol::Real=1e-8) where {T<:AbstractFloat}

Return the numerical column rank and a pivot vector.

The rank is determined from the absolute values of the diagonal of R from
a pivoted QR decomposition, relative to the first (and, hence, largest) 
element of this vector.

In the full-rank case the pivot vector is `collect(axes(x, 2))`.
"""
function statsrank(x::AbstractMatrix{T}; ranktol=1e-8) where {T<:AbstractFloat}
    m, n = size(x)

    qrpiv = qr(x, Val(true))
    dvec = abs.(diag(qrpiv.R))
    fdv = first(dvec)
    if (last(dvec) / fdv) > ranktol
        return (rank = n, piv = collect(axes(x, 2)))
    end
    rank = searchsortedlast(dvec ./ fdv, ranktol, rev=true)
    @assert rank < n
    piv = qrpiv.p
    v1 = first(eachcol(x))
    if all(isone, v1) && first(piv) ≠ 1
        # make sure the first column isn't moved by inflating v1
        v1 .*= (fdv + true) / sqrt(m)
        qrpiv = qr(x, Val(true))
        piv = qrpiv.p
        fill!(v1, one(T))    # restore the contents of the first column
    end

    # preserve as much of the original column ordering as possible
    sort!(view(piv, 1:rank)) # maintain the original column order for the linearly independent columns
    (rank = rank, piv = piv)
end
