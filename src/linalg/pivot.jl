"""
    statscholesky(xtx::Symmetric{T}, tol::Real=-1) where {T<:AbstractFloat}
Return a `CholeskyPivoted` object created from `xtx` where the pivoting scheme
retains the original order unless singularity is detected.  Columns that are
(computationally) linearly dependent on columns to their left are moved to the
right hand side in a left circular shift.
"""
function statscholesky(xtx::Symmetric{T}, tol::Real = -1) where {T<:AbstractFloat}
    n = size(xtx, 2)
    chpiv = cholesky(xtx, Val(true), tol = T(tol), check = false)
    chunp = cholesky(xtx, check = false);

    piv = [1:n;]
    r = chpiv.rank

    if r < n

        k = chunp.info
        if k > r
            @warn """Fixed-effects matrix may be ill conditioned.
                     Left-circular shift may fail."""
        end

        nleft = n
        while nleft > r
            # the 0 lowerbound is for MKL compatibility
            # this arises when the unpivoted Cholesky succeeds but the pivoted
            # Chokesky estimates less than full rank (see #367)
            if 0 < k < nleft
                piv = piv[[1:k-1; k+1:n; k]]
                chunp = cholesky!(Symmetric(xtx[piv, piv]), check = false)
            end
            k = chunp.info
            nleft -= 1
        end

        for j = (r+1):n   # an MKL <-> OpenBLAS difference
            for i = (r+1):j
                chunp.factors[i, j] = zero(T)
            end
        end

    end

    CholeskyPivoted(chunp.factors, chunp.uplo, piv, r, tol, chpiv.info)
end

"""
    statsqr(x::Matrix{T}, ranktol::Real=1e-8) where {T<:AbstractFloat}
Return a `QRPivoted` object created from `x` where the pivoting scheme
retains the original order unless singularity is detected.
"""
function statsqr(x::Matrix{T}; ranktol=1e-8) where {T<:AbstractFloat}
    n = size(x, 2)

    qrpiv = qr(x, Val(true))
    rank =  searchsortedlast(abs.(diag(qrpiv.R)), ranktol, rev=true);
    if rank < n
        piv = qrpiv.p
    else
        piv = collect(1:n)
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