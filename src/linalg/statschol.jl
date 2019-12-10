"""
    statscholesky(xtx::Symmetric{T}, tol::Real=-1) where {T<:AbstractFloat}

Return a `CholeskyPivoted` object created from `xtx` where the pivoting scheme
retains the original order unless singularity is detected.  Columns that are
(computationally) linearly dependent on columns to their left are moved to the
right hand side in a left circular shift.
"""
function statscholesky(xtx::Symmetric{T}, tol::Real = -1) where {T<:AbstractFloat}
    n = size(xtx, 2)
    chpiv = cholesky(xtx, Val(true), tol = T(-1), check = false)
    chunp = cholesky(xtx, check = false)
    r = chpiv.rank
    piv = [1:n;]
    if r < n
        nleft = n
        while r < nleft
            k = chunp.info
            if k < nleft
                piv = piv[[1:k-1; k+1:n; k]]
                chunp = cholesky!(Symmetric(xtx[piv, piv]), check = false)
            end
            nleft -= 1
        end
    end
    for j = (r+1):n   # an MKL <-> OpenBLAS difference
        for i = (r+1):j
            chunp.factors[i, j] = zero(T)
        end
    end
    CholeskyPivoted(chunp.factors, chunp.uplo, piv, r, tol, chpiv.info)
end
