"""
    statsqr(x::Matrix{T}, ranktol::Real=1e-8) where {T<:AbstractFloat}

Return the numerical column rank and, for a rank-deficient X, a pivot vector
"""
function statsqr(x::Matrix{T}; ranktol=1e-8) where {T<:AbstractFloat}
    n = size(x, 2)

    qrpiv = qr(x, Val(true))
    rank =  searchsortedlast(abs.(diag(qrpiv.R)), ranktol, rev=true);
    if rank < n
        piv = qrpiv.p
    else
        piv = LinearAlgebra.BlasInt.(collect(1:n))
    end

    if rank < n && piv[1] ≠ 1
        # make sure the first column isn't moved
        # this is usually the intercept and it's desirable to avoid
        # pivoting the intercept out. note that this inflation may
        # not always be enough, but if it isn't, then you probably
        # have other problems

        x = deepcopy(x) # this is horrible
        x[:, 1] .*= 1e6
        qrpiv = qr(x, Val(true))
        piv = qrpiv.p
    end

    # preserve as much of the original column ordering as possible
    piv = [sort!(piv[1:rank]); sort!(piv[rank+1:n])];
    qrunp = qr(x[:, piv], Val(false));

    QRPivoted(qrunp.factors, qrpiv.τ, piv)
end