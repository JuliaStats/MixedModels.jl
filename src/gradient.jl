# Evaluate analytic gradient of the objective for ML or REML fitting of a LinearMixedModel
#
# The objective is an affine function of the logarithms of the diagonal elements of the
# blocked lower Cholesky factor L,
#
#   obj = 2 Σⱼ wⱼ log Lⱼⱼ + constant,
#
# with weights wⱼ = 1 on the random-effects rows, wⱼ = 1 (REML) or 0 (ML) on the
# fixed-effects rows, and w on the last row (the ℓ_yy element) equal to n (ML), n - p
# (REML), or pwrss / σ² when σ is fixed.  Differentiating both sides of Ω = L Lᵀ
# (Murray 2016, arXiv:1602.07527) gives diag(L⁻¹ Ω̇ L⁻ᵀ)ⱼⱼ = 2 ∂log Lⱼⱼ, so
#
#   ∂obj/∂θₚ = tr(W L⁻¹ Ω̇ₚ L⁻ᵀ) = ⟨S, Ω̇ₚ⟩,   S = L⁻ᵀ W L⁻¹,
#
# where W = diag(w).  S does not depend on p, so a single blocked computation of
# X = L⁻¹ (lower blocks only) provides, through weighted Gram products S = Xᵀ W X,
# every component of the gradient.  For the parameter θₚ ↦ λ_b[i,j] (`parmap[p] = (b,i,j)`)
# the derivative Ω̇ₚ is supported on block row/column b and
#
#   ∂obj/∂θₚ = 2 G_b[i,j],   G_b = Σ_faces (Λᵀ A)[:, block-col b]ᵀ (S E_b)
#
# accumulated face-by-face over the levels of the grouping factor, which is evaluated
# blockwise against the sparse structure of the A blocks without ever forming Ω̇ₚ.

"""
    GradientWorkspace(m::LinearMixedModel)

Preallocated storage for evaluating the gradient of the objective of `m`.

The workspace holds the lower blocks of `X = L⁻¹` (`X[r,c]`, `r ≥ c`), buffers for the
blocks of `S = XᵀWX` that are contracted against the corresponding `A` blocks, scratch
copies of off-diagonal `A` blocks premultiplied by `Λᵣᵀ` (`C1`) or postmultiplied by
`Λ_b` (`C2`), and one `k_b × k_b` accumulator `G_b` per random-effects term.

For blocks between two scalar terms where `A[r,b]` is sparse, only the entries of `S`
matching the sparsity pattern are evaluated and no buffer is stored.
"""
# column-panel width for the BLAS-3 evaluation of the cross term between two scalar
# terms whose Cholesky fill block is dense (see `_crosspair_blas3!`)
const GRAD_PANEL = 128

struct GradientWorkspace{T<:AbstractFloat}
    X::Matrix{AbstractMatrix{T}}   # lower blocks of L⁻¹; upper cells are 0×0 placeholders
    S::Matrix{AbstractMatrix{T}}   # per-pair buffers for blocks of S = XᵀWX
    C1::Matrix{AbstractMatrix{T}}  # Λᵣᵀ A[r,b] for r > b (dense pairs only)
    C2::Matrix{AbstractMatrix{T}}  # A[r,b] Λ_b for r > b (dense pairs only)
    G::Vector{Matrix{T}}           # per-term gradient accumulators (k_b × k_b)
    Ppanel::Matrix{T}              # q_r × GRAD_PANEL scratch for the BLAS-3 cross term
end

_kdim(rt::ReMat{T,S}) where {T,S} = S

function GradientWorkspace(m::LinearMixedModel{T}) where {T}
    (; A, L, reterms) = m
    k = length(reterms)
    nb = k + 1
    placeholder = Matrix{T}(undef, 0, 0)
    X = fill!(Matrix{AbstractMatrix{T}}(undef, nb, nb), placeholder)
    S = fill!(Matrix{AbstractMatrix{T}}(undef, nb, k), placeholder)
    C1 = fill!(Matrix{AbstractMatrix{T}}(undef, nb, k), placeholder)
    C2 = fill!(Matrix{AbstractMatrix{T}}(undef, nb, k), placeholder)
    for c in 1:nb
        Lcc = L[kp1choose2(c)]
        X[c, c] = if isa(Lcc, Diagonal)
            Diagonal(Vector{T}(undef, size(Lcc, 1)))
        else
            Matrix{T}(undef, size(Lcc))
        end
        for r in (c + 1):nb
            X[r, c] = Matrix{T}(undef, size(L[block(r, c)]))
        end
    end
    maxheavy = 0
    for b in 1:k
        Abb = A[kp1choose2(b)]
        if isa(Abb, Diagonal)
            S[b, b] = Diagonal(Vector{T}(undef, size(Abb, 1)))
        else    # UniformBlockDiagonal
            S[b, b] = UniformBlockDiagonal(Array{T,3}(undef, size(Abb.data)))
        end
        S[nb, b] = Matrix{T}(undef, size(A[block(nb, b)]))
        for r in (b + 1):k
            Arb = A[block(r, b)]
            if _sparsepair(Arb, reterms[r], reterms[b])
                # scalar-scalar pair: entries of S accumulated directly, no S/C buffer.
                # When the fill block L[r,r] is dense the cross term is evaluated with a
                # BLAS-3 kernel needing a q_r × GRAD_PANEL scratch (see `_crosspair_blas3!`)
                isa(L[kp1choose2(r)], Matrix) && (maxheavy = max(maxheavy, size(Arb, 1)))
                continue
            end
            S[r, b] = Matrix{T}(undef, size(Arb))
            C1[r, b] = Matrix{T}(undef, size(Arb))
            C2[r, b] = Matrix{T}(undef, size(Arb))
        end
    end
    G = [Matrix{T}(undef, _kdim(rt), _kdim(rt)) for rt in reterms]
    Ppanel = Matrix{T}(undef, maxheavy, iszero(maxheavy) ? 0 : GRAD_PANEL)
    return GradientWorkspace{T}(X, S, C1, C2, G, Ppanel)
end

# sparse selected-entry path is available only between two scalar terms
function _sparsepair(A::AbstractMatrix, rtr::AbstractReMat, rtb::AbstractReMat)
    return (isa(A, SparseMatrixCSC) || isa(A, BlockedSparse)) &&
           isone(_kdim(rtr)) && isone(_kdim(rtb))
end

_cscmat(A::BlockedSparse) = A.cscmat
_cscmat(A::SparseMatrixCSC) = A
_densemat(A::AbstractMatrix) = A
_densemat(A::BlockedSparse) = A.cscmat

# mul!(C, A, B, -1, 1) with the block types that occur in L and X
function _mulsub!(
    C::AbstractMatrix{T}, A::AbstractMatrix{T}, B::AbstractMatrix{T}
) where {T}
    return mul!(C, A, B, -one(T), one(T))
end

function _mulsub!(C::AbstractMatrix{T}, A::BlockedSparse{T}, B::AbstractMatrix{T}) where {T}
    return _mulsub!(C, A.cscmat, B)
end

function _mulsub!(C::Matrix{T}, A::SparseMatrixCSC{T}, B::Diagonal{T}) where {T}
    Bd = B.diag
    rv = rowvals(A)
    nz = nonzeros(A)
    @inbounds for j in axes(A, 2)
        d = Bd[j]
        for idx in nzrange(A, j)
            C[rv[idx], j] -= nz[idx] * d
        end
    end
    return C
end

# in-place solve of Ljj \ B for the diagonal-block types of L
_ldivL!(Ljj::Diagonal{T}, B::AbstractMatrix{T}) where {T} = ldiv!(Ljj, B)
_ldivL!(Ljj::Matrix{T}, B::AbstractMatrix{T}) where {T} = ldiv!(LowerTriangular(Ljj), B)
function _ldivL!(Ljj::UniformBlockDiagonal{T}, B::Matrix{T}) where {T}
    return ldiv!(LowerTriangular(Ljj), B)
end

_identity!(D::Diagonal{T}) where {T} = (fill!(D.diag, one(T)); D)
function _identity!(X::Matrix{T}) where {T}
    fill!(X, zero(T))
    @inbounds for i in diagind(X)
        X[i] = one(T)
    end
    return X
end

"""
    _invL!(w::GradientWorkspace{T}, m::LinearMixedModel{T})

Overwrite the lower blocks of `w.X` with the corresponding blocks of `L⁻¹`.
"""
function _invL!(w::GradientWorkspace{T}, m::LinearMixedModel{T}) where {T}
    L = m.L
    X = w.X
    nb = size(X, 1)
    for c in 1:nb
        Xcc = _identity!(X[c, c])
        _ldivL!(L[kp1choose2(c)], Xcc)
        for r in (c + 1):nb
            Xrc = fill!(X[r, c], zero(T))
            for s in c:(r - 1)
                _mulsub!(Xrc, L[block(r, s)], X[s, c])
            end
            _ldivL!(L[kp1choose2(r)], Xrc)
        end
    end
    return w
end

# weight on the last diagonal element of L (the ℓ_yy element)
function _yweight(m::LinearMixedModel{T}) where {T}
    σ = m.optsum.sigma
    return isnothing(σ) ? T(ssqdenom(m)) : pwrss(m) / T(σ)^2
end

#####
##### weighted Gram products: blocks of S = Xᵀ W X
#####

# accumulate the [Xy]-block-row correction Xkrᵀ W Xkb into S, where W has weight
# wx on the first p rows and wy on the last row
function _xycorrection!(
    S::Matrix{T}, Xkr::Matrix{T}, Xkb::Matrix{T}, wx::T, wy::T
) where {T}
    plast = size(Xkr, 1)
    if !iszero(wx)
        mul!(S, Xkr', Xkb, wx, one(T))
        iszero(wy - wx) && return S
        wy = wy - wx    # adjust the last-row weight for the part already added
    end
    xr = view(Xkr, plast, :)
    xb = view(Xkb, plast, :)
    return mul!(S, xr, xb', wy, one(T))
end

# dense S block for the pair (r, b), r > b or r == nb (the [Xy] row)
function _gram!(S::Matrix{T}, w::GradientWorkspace{T}, r::Int, b::Int, kre::Int,
    wx::T, wy::T) where {T}
    X = w.X
    fill!(S, zero(T))
    for s in r:kre
        mul!(S, X[s, r]', X[s, b], one(T), one(T))
    end
    return _xycorrection!(S, X[kre + 1, r], X[kre + 1, b], wx, wy)
end

# diagonal of S[b,b] (used when A[b,b] is Diagonal, i.e. a scalar term)
_colsumabs2!(d::Vector{T}, X::Diagonal{T}) where {T} = (d .+= abs2.(X.diag); d)

function _colsumabs2!(d::Vector{T}, X::Matrix{T}) where {T}
    @inbounds for f in axes(X, 2)
        acc = zero(T)
        for i in axes(X, 1)
            acc += abs2(X[i, f])
        end
        d[f] += acc
    end
    return d
end

function _gramdiag!(S::Diagonal{T}, w::GradientWorkspace{T}, b::Int, kre::Int,
    wx::T, wy::T) where {T}
    d = fill!(S.diag, zero(T))
    for s in b:kre
        _colsumabs2!(d, w.X[s, b])
    end
    Xkb = w.X[kre + 1, b]::Matrix{T}
    plast = size(Xkb, 1)
    @inbounds for f in axes(Xkb, 2)
        acc = zero(T)
        if !iszero(wx)
            for i in 1:(plast - 1)
                acc += wx * abs2(Xkb[i, f])
            end
        end
        d[f] += acc + wy * abs2(Xkb[plast, f])
    end
    return S
end

# face-diagonal blocks of S[b,b] (used when A[b,b] is UniformBlockDiagonal)
function _gramfaces!(S::UniformBlockDiagonal{T}, w::GradientWorkspace{T}, b::Int,
    kre::Int, wx::T, wy::T) where {T}
    dat = fill!(S.data, zero(T))
    kb = size(dat, 1)
    for s in b:kre
        Xsb = w.X[s, b]::Matrix{T}
        for f in axes(dat, 3)
            cols = ((f - 1) * kb + 1):(f * kb)
            Xv = view(Xsb, :, cols)
            mul!(view(dat, :, :, f), Xv', Xv, one(T), one(T))
        end
    end
    Xkb = w.X[kre + 1, b]::Matrix{T}
    plast = size(Xkb, 1)
    @inbounds for f in axes(dat, 3)
        coloffset = (f - 1) * kb
        for c in 1:kb, a in 1:kb
            acc = zero(T)
            if !iszero(wx)
                for i in 1:(plast - 1)
                    acc += wx * Xkb[i, coloffset + a] * Xkb[i, coloffset + c]
                end
            end
            acc += wy * Xkb[plast, coloffset + a] * Xkb[plast, coloffset + c]
            dat[a, c, f] += acc
        end
    end
    return S
end

# Σ over the nonzeros (u, v) of A of A[u,v] * (Xrᵀ Xb)[u,v] for one block-row pair of X.
# These methods are function barriers: the X blocks are stored with an abstract element
# type and the entry loops must run with concretely typed arrays.
function _sparseacc(A::SparseMatrixCSC{T}, Xr::Diagonal{T}, Xb::Matrix{T}) where {T}
    rv = rowvals(A)
    nz = nonzeros(A)
    d = Xr.diag
    acc = zero(T)
    @inbounds for v in axes(A, 2)
        for idx in nzrange(A, v)
            u = rv[idx]
            acc += nz[idx] * d[u] * Xb[u, v]
        end
    end
    return acc
end

function _sparseacc(A::SparseMatrixCSC{T}, Xr::Matrix{T}, Xb::Matrix{T}) where {T}
    rv = rowvals(A)
    nz = nonzeros(A)
    acc = zero(T)
    @inbounds for v in axes(A, 2)
        xbv = view(Xb, :, v)
        for idx in nzrange(A, v)
            acc += nz[idx] * dot(view(Xr, :, rv[idx]), xbv)
        end
    end
    return acc
end

# ditto for the [Xy] block row with weight wx on the first p rows and wy on the last
function _sparseaccxy(A::SparseMatrixCSC{T}, Xkr::Matrix{T}, Xkb::Matrix{T},
    wx::T, wy::T) where {T}
    rv = rowvals(A)
    nz = nonzeros(A)
    plast = size(Xkr, 1)
    acc = zero(T)
    @inbounds for v in axes(A, 2)
        for idx in nzrange(A, v)
            u = rv[idx]
            s = wy * Xkr[plast, u] * Xkb[plast, v]
            if !iszero(wx)
                for i in 1:(plast - 1)
                    s += wx * Xkr[i, u] * Xkb[i, v]
                end
            end
            acc += nz[idx] * s
        end
    end
    return acc
end

#####
##### contraction of A blocks against S blocks into the per-term accumulators G
#####

# accumulate G += Σ_faces C[:, face]ᵀ S[:, face] where the faces are the
# kb-column groups of the term-b block column
function _facecontract!(G::Matrix{T}, C::Matrix{T}, S::Matrix{T}) where {T}
    kb = size(G, 1)
    for f in 1:(size(C, 2) ÷ kb)
        cols = ((f - 1) * kb + 1):(f * kb)
        mul!(G, view(C, :, cols)', view(S, :, cols), one(T), one(T))
    end
    return G
end

# accumulate G += Σ_faces C[face rows, :] S[face rows, :]ᵀ where the faces are the
# kr-row groups of the term-r block row
function _facecontract_rows!(G::Matrix{T}, C::Matrix{T}, S::Matrix{T}) where {T}
    kr = size(G, 1)
    for f in 1:(size(C, 1) ÷ kr)
        rows = ((f - 1) * kr + 1):(f * kr)
        mul!(G, view(C, rows, :), view(S, rows, :)', one(T), one(T))
    end
    return G
end

# diagonal pair (b, b)
function _diagpair!(w::GradientWorkspace{T}, m::LinearMixedModel{T}, b::Int,
    wx::T, wy::T) where {T}
    (; A, reterms) = m
    kre = length(reterms)
    Abb = A[kp1choose2(b)]
    G = w.G[b]
    if isa(Abb, Diagonal)
        S = _gramdiag!(w.S[b, b]::Diagonal{T}, w, b, kre, wx, wy)
        G[1, 1] += T(only(reterms[b].λ)) * dot(Abb.diag, S.diag)
    else
        Abb = Abb::UniformBlockDiagonal{T}
        S = _gramfaces!(w.S[b, b]::UniformBlockDiagonal{T}, w, b, kre, wx, wy)
        λ = reterms[b].λ
        kb = size(G, 1)
        t = Matrix{T}(undef, kb, kb)
        for f in axes(S.data, 3)
            mul!(t, λ, view(S.data, :, :, f))
            mul!(G, adjoint(view(Abb.data, :, :, f)), t, one(T), one(T))
        end
    end
    return w
end

# off-diagonal pair (r, b) between two scalar terms with sparse A[r,b]:
# only the entries of S on the sparsity pattern of A are evaluated
function _sparsepair!(w::GradientWorkspace{T}, m::LinearMixedModel{T}, r::Int, b::Int,
    wx::T, wy::T) where {T}
    (; A, reterms) = m
    kre = length(reterms)
    Arb = _cscmat(A[block(r, b)])
    acc = zero(T)
    for s in r:kre
        acc += _sparseacc(Arb, w.X[s, r], w.X[s, b]::Matrix{T})
    end
    acc += _sparseaccxy(Arb, w.X[kre + 1, r]::Matrix{T}, w.X[kre + 1, b]::Matrix{T}, wx, wy)
    w.G[b][1, 1] += T(only(reterms[r].λ)) * acc
    w.G[r][1, 1] += T(only(reterms[b].λ)) * acc
    return w
end

# ⟨A, Xᵣᵣᵀ Xᵣᵦ⟩ over the nonzeros of A, evaluated a column-panel at a time.  The dense
# fill block Xᵣᵦ (and the dense inverse block Xᵣᵣ) make the s = r term of the scalar-scalar
# cross product the dominant cost; a BLAS-3 matrix product per panel replaces the many
# BLAS-1 column dot products of `_sparseacc`, which is memory-bandwidth bound.  Only a
# `q_r × GRAD_PANEL` slice of the product is materialized at a time.
function _crossacc_blas3!(
    Pp::Matrix{T}, A::SparseMatrixCSC{T}, Xrr::Matrix{T}, Xrb::Matrix{T}
) where {T}
    rv = rowvals(A)
    nz = nonzeros(A)
    qr = size(Xrr, 2)
    qb = size(Xrb, 2)
    acc = zero(T)
    coloff = 0
    while coloff < qb
        width = min(size(Pp, 2), qb - coloff)
        Pv = view(Pp, 1:qr, 1:width)
        mul!(Pv, Xrr', view(Xrb, :, (coloff + 1):(coloff + width)))
        @inbounds for j in 1:width
            for idx in nzrange(A, coloff + j)
                acc += nz[idx] * Pv[rv[idx], j]
            end
        end
        coloff += width
    end
    return acc
end

# is the BLAS-3 cross-term path worthwhile for pair (r, b)?  It needs a dense fill block
# and pays off only when A[r,b] is dense enough that the extra flops of the full product
# are outweighed by BLAS-3 throughput (the crossover ratio ≈ rate_BLAS1 / rate_BLAS3)
function _use_blas3_cross(w::GradientWorkspace, m::LinearMixedModel, r::Int, b::Int)
    isempty(w.Ppanel) && return false
    isa(w.X[r, r], Matrix) || return false
    Arb = _cscmat(m.A[block(r, b)])
    return nnz(Arb) > 0.03 * size(Arb, 1) * size(Arb, 2)
end

# same result as `_sparsepair!`, but the heavy s = r term uses the BLAS-3 kernel
function _crosspair_blas3!(w::GradientWorkspace{T}, m::LinearMixedModel{T}, r::Int, b::Int,
    wx::T, wy::T) where {T}
    (; A, reterms) = m
    kre = length(reterms)
    Arb = _cscmat(A[block(r, b)])
    acc = _crossacc_blas3!(w.Ppanel, Arb, w.X[r, r]::Matrix{T}, w.X[r, b]::Matrix{T})
    for s in (r + 1):kre    # remaining (light) blocks stay on the sparse BLAS-1 path
        acc += _sparseacc(Arb, w.X[s, r], w.X[s, b]::Matrix{T})
    end
    acc += _sparseaccxy(Arb, w.X[kre + 1, r]::Matrix{T}, w.X[kre + 1, b]::Matrix{T}, wx, wy)
    w.G[b][1, 1] += T(only(reterms[r].λ)) * acc
    w.G[r][1, 1] += T(only(reterms[b].λ)) * acc
    return w
end

# off-diagonal pair (r, b), r > b, dense path: contract against C1 = Λᵣᵀ A[r,b]
# (for G_b) and C2 = A[r,b] Λ_b (for G_r)
function _densepair!(w::GradientWorkspace{T}, m::LinearMixedModel{T}, r::Int, b::Int,
    wx::T, wy::T) where {T}
    (; A, reterms) = m
    kre = length(reterms)
    Arb = _densemat(A[block(r, b)])
    S = _gram!(w.S[r, b]::Matrix{T}, w, r, b, kre, wx, wy)
    C1 = copyto!(w.C1[r, b]::Matrix{T}, Arb)
    lmulΛ!(reterms[r]', C1)
    _facecontract!(w.G[b], C1, S)
    C2 = copyto!(w.C2[r, b]::Matrix{T}, Arb)
    rmulΛ!(C2, reterms[b])
    _facecontract_rows!(w.G[r], C2, S)
    return w
end

# the [Xy] block row (r = k + 1): no Λ factor
function _xypair!(w::GradientWorkspace{T}, m::LinearMixedModel{T}, b::Int,
    wx::T, wy::T) where {T}
    kre = length(m.reterms)
    nb = kre + 1
    Akb = m.A[block(nb, b)]::Matrix{T}
    S = _gram!(w.S[nb, b]::Matrix{T}, w, nb, b, kre, wx, wy)
    return _facecontract!(w.G[b], Akb, S)
end

"""
    objective_gradient!(g::AbstractVector{T}, m::LinearMixedModel{T})
    objective_gradient!(g::AbstractVector{T}, m::LinearMixedModel{T}, θ::AbstractVector{T})

Overwrite `g` with the gradient of the [`objective`](@ref) (negative twice the profiled
log-likelihood, or the REML criterion when `m.optsum.REML` is set) with respect to the
covariance parameters θ, and return the value of the objective.

The three-argument method installs `θ` via [`setθ!`](@ref) and [`updateL!`](@ref) first;
the two-argument method evaluates at the current parameter values of `m` (which must have
an up-to-date `L`, e.g. from a previous call to `updateL!`).

The gradient is evaluated analytically from the blocked Cholesky factor: the objective is
an affine function of the logarithms of the diagonal elements of `L` and a single blocked
computation of `L⁻¹` provides all components of the gradient (see Murray 2016,
arXiv:1602.07527, and Bates et al. 2025, arXiv:2505.11674).  This is much faster and less
allocation-heavy than automatic differentiation via the `ForwardDiff` extension,
especially for models with many covariance parameters, but note that the storage for the
blocks of `L⁻¹` can be substantial for models with thousands of random-effects levels.
"""
function objective_gradient!(g::AbstractVector{T}, m::LinearMixedModel{T}) where {T}
    return objective_gradient!(GradientWorkspace(m), g, m)
end

function objective_gradient!(g::AbstractVector{T}, m::LinearMixedModel{T},
    θ::AbstractVector{T}) where {T}
    return objective_gradient!(g, updateL!(setθ!(m, θ)))
end

function objective_gradient!(w::GradientWorkspace{T}, g::AbstractVector{T},
    m::LinearMixedModel{T}, θ::AbstractVector{T}) where {T}
    return objective_gradient!(w, g, updateL!(setθ!(m, θ)))
end

function objective_gradient!(w::GradientWorkspace{T}, g::AbstractVector{T},
    m::LinearMixedModel{T}) where {T}
    (; parmap, reterms, optsum) = m
    if length(g) ≠ length(parmap)
        throw(DimensionMismatch("length(g) = $(length(g)) should be $(length(parmap))"))
    end
    kre = length(reterms)
    wx = optsum.REML ? one(T) : zero(T)
    wy = _yweight(m)
    _invL!(w, m)
    for G in w.G
        fill!(G, zero(T))
    end
    for b in 1:kre
        _diagpair!(w, m, b, wx, wy)
        for r in (b + 1):kre
            if _sparsepair(m.A[block(r, b)], reterms[r], reterms[b])
                if _use_blas3_cross(w, m, r, b)
                    _crosspair_blas3!(w, m, r, b, wx, wy)
                else
                    _sparsepair!(w, m, r, b, wx, wy)
                end
            else
                _densepair!(w, m, r, b, wx, wy)
            end
        end
        _xypair!(w, m, b, wx, wy)
    end
    for (p, (b, i, j)) in enumerate(parmap)
        g[p] = 2 * w.G[b][i, j]
    end
    return objective(m)
end
# in-place blocked solve of a UniformBlockDiagonal lower-triangular system with a
# dense right-hand side, used by `_ldivL!`
function LinearAlgebra.ldiv!(
    A::LowerTriangular{T,UniformBlockDiagonal{T}},
    B::Matrix{T},
) where {T}
    if size(A, 2) ≠ size(B, 1)
        throw(DimensionMismatch("size(A,2) = $(size(A,2)) ≠ $(size(B,1)) = size(B,1)"))
    end
    A_dat = A.data.data
    axis1 = axes(A_dat, 1)
    offset = 0
    for k in axes(A_dat, 3)
        ldiv!(LowerTriangular(view(A_dat, :, :, k)), view(B, offset .+ axis1, :))
        offset += length(axis1)
    end
    return B
end
