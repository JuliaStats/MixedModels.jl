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
