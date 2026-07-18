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
#
# The blocks of X mirror the structure of the corresponding blocks of L wherever the
# inverse preserves that structure: block-diagonal (`Diagonal`, `UniformBlockDiagonal`)
# diagonal blocks have block-diagonal inverses, and an off-diagonal `BlockedSparse` block
# of L propagates its sparsity pattern to X when the diagonal blocks above it are
# block-diagonal (the nested-grouping-factor case).  Only the entries of S matching the
# sparsity of the A blocks are ever evaluated for such pairs, so the workspace memory is
# of the same order as the storage for L itself.

"""
    GradientWorkspace(m::LinearMixedModel)

Preallocated storage for evaluating the gradient of the objective of `m`.

The workspace holds the lower blocks of `X = L⁻¹` (`X[r,c]`, `r ≥ c`), buffers for the
blocks of `S = XᵀWX` that are contracted against the corresponding `A` blocks, scratch
copies of off-diagonal `A` blocks premultiplied by `Λᵣᵀ` (`C1`) or postmultiplied by
`Λ_b` (`C2`, dense pairs only), and one `k_b × k_b` accumulator `G_b` per
random-effects term.

The blocks of `X` mirror the structure of the corresponding blocks of `L`:
block-diagonal diagonal blocks stay block-diagonal, a diagonal block stored in
rectangular full packed format (`TriangularRFP`, see the `RFPthreshold` argument of
[`LinearMixedModel`](@ref)) is inverted in the same packed storage, and `BlockedSparse`
off-diagonal blocks (nested grouping factors) are stored as `SparseMatrixCSC` sharing
the pattern of the `L` block.  For pairs whose `A` block is sparse, only the entries of
`S` matching the sparsity pattern of `A` are evaluated: between two scalar terms they are
accumulated directly without a buffer, otherwise into a `SparseMatrixCSC` buffer
mirroring the pattern of `A`.  Dense `S`/`C1`/`C2` buffers are allocated only for pairs
whose `A` block is dense.
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
    path::Matrix{Symbol}           # `_pairpath` route per off-diagonal pair (r, b)
end

_kdim(rt::ReMat{T,S}) where {T,S} = S

_cscmat(A::BlockedSparse) = A.cscmat
_cscmat(A::SparseMatrixCSC) = A
_densemat(A::AbstractMatrix) = A
_densemat(A::BlockedSparse) = A.cscmat

# the working part of a diagonal block of L: `createAL` wraps dense and
# block-diagonal diagonal blocks in LowerTriangular, while Diagonal,
# TriangularRFP, and the trailing [Xy] block are stored bare
_diagdata(A::AbstractMatrix) = A
_diagdata(A::LowerTriangular) = parent(A)

# views of the two dense regions of a (transr = 'N', uplo = 'L') TriangularRFP of
# order n: with L = [L11 0; L21 L22], L11 of order m1 = n - n ÷ 2 and L22 of order
# m2 = n ÷ 2, `t` holds the trapezoid [L11; L21] (only the lower triangle of its
# first m1 rows is meaningful) and `u` holds L22' as its upper triangle (including
# the diagonal); entry (i, j) of L is t[i, j] for j ≤ m1 and u[j - m1, i - m1] else
function _rfpviews(A::TriangularRFP)
    (A.transr == 'N' && A.uplo == 'L') ||
        throw(ArgumentError("A must be in transr = 'N', uplo = 'L' storage"))
    dat = A.data
    n = size(A, 1)
    m2 = n >> 1
    m1 = n - m2
    shift = iseven(n)   # the trapezoid starts on row 2 of the parent for even n
    t = view(dat, (1 + shift):(n + shift), 1:m1)
    u = view(dat, 1:m2, (1 + m1 - m2):m1)
    return t, u, m1, m2
end

# a SparseMatrixCSC sharing the pattern (colptr, rowval) of A, with its own nzval.
# The pattern arrays are never mutated through either matrix.
function _patternmirror(A::SparseMatrixCSC{T,Ti}) where {T,Ti}
    m, n = size(A)
    return SparseMatrixCSC{T,Ti}(m, n, A.colptr, A.rowval, Vector{T}(undef, nnz(A)))
end

# do the nonzeros of A form complete kr-row runs starting on kr-row boundaries, with
# all kb columns of each kb-column block sharing the same row pattern?  This is the
# layout produced by products of `ReMat`s (cf. the reshapes in `rmulΛ!` and `rdiv!`
# for `BlockedSparse`) and is required by the blockwise kernels below.
function _blockaligned(A::SparseMatrixCSC, kr::Integer, kb::Integer)
    iszero(size(A, 2) % kb) || return false
    rv = rowvals(A)
    for jblk in 1:(size(A, 2) ÷ kb)
        v1 = (jblk - 1) * kb + 1
        rng1 = nzrange(A, v1)
        iszero(length(rng1) % kr) || return false
        for i in first(rng1):kr:last(rng1)
            iszero((rv[i] - 1) % kr) || return false
            for l in 1:(kr - 1)
                rv[i + l] == rv[i] + l || return false
            end
        end
        for j in (v1 + 1):(v1 + kb - 1)
            rng = nzrange(A, j)
            length(rng) == length(rng1) || return false
            for (i, i1) in zip(rng, rng1)
                rv[i] == rv[i1] || return false
            end
        end
    end
    return true
end

# is pattern(A * B) contained in pattern(C)?  (all matrices column-sorted CSC)
function _productcontained(
    A::SparseMatrixCSC, B::SparseMatrixCSC, C::SparseMatrixCSC
)
    Arv = rowvals(A)
    Brv = rowvals(B)
    Crv = rowvals(C)
    for v in axes(B, 2)
        crng = nzrange(C, v)
        for bidx in nzrange(B, v)
            w = Brv[bidx]
            ci = first(crng)
            for aidx in nzrange(A, w)
                u = Arv[aidx]
                while ci ≤ last(crng) && Crv[ci] < u
                    ci += 1
                end
                (ci ≤ last(crng) && Crv[ci] == u) || return false
            end
        end
    end
    return true
end

# can X[r,c] be stored as a sparse mirror of L[r,c]?  Requires a block-diagonal
# inverse of L[r,r] (so the row pattern is preserved), block-aligned columns, and,
# for intermediate terms s, that the products L[r,s]·X[s,c] stay within the pattern.
# Reads X[c,c] and X[s,c] for s < r, so the workspace constructor must fill each
# block column of X in increasing row order before classifying X[r,c]
function _sparseXok(L, X::Matrix{AbstractMatrix{T}}, reterms, r::Int, c::Int) where {T}
    Lrc = L[block(r, c)]
    isa(Lrc, Union{BlockedSparse{T},SparseMatrixCSC{T}}) || return false
    isa(_diagdata(L[kp1choose2(r)]), Union{Diagonal{T},UniformBlockDiagonal{T}}) ||
        return false
    isa(X[c, c], Union{Diagonal{T},UniformBlockDiagonal{T}}) || return false
    Lcsc = _cscmat(Lrc)
    _blockaligned(Lcsc, _kdim(reterms[r]), _kdim(reterms[c])) || return false
    for s in (c + 1):(r - 1)
        Xsc = X[s, c]
        isa(Xsc, SparseMatrixCSC{T}) || return false
        Lrs = L[block(r, s)]
        isa(Lrs, Union{BlockedSparse{T},SparseMatrixCSC{T}}) || return false
        _productcontained(_cscmat(Lrs), Xsc, Lcsc) || return false
    end
    return true
end

# how is the off-diagonal pair (r, b) evaluated?
#   :scalar   - scalar-scalar with all-dense X blocks: entries of S accumulated
#               directly on the pattern of A (`_sparsepair!` / `_crosspair_blas3!`)
#   :selected - sparse A block: entries of S on the pattern of A accumulated into a
#               sparse buffer and contracted blockwise (`_selectedpair!`)
#   :dense    - dense A block: dense S/C1/C2 buffers (`_densepair!`)
function _pairpath(
    Arb::AbstractMatrix{T}, X::Matrix{AbstractMatrix{T}}, reterms, r::Int, b::Int
) where {T}
    isa(Arb, Union{SparseMatrixCSC{T},BlockedSparse{T}}) || return :dense
    kre = length(reterms)
    if isone(_kdim(reterms[r])) &&
        isone(_kdim(reterms[b])) &&
        isa(X[r, b], Matrix{T}) &&
        all(isa(X[s, r], Matrix{T}) && isa(X[s, b], Matrix{T}) for s in (r + 1):kre)
        return :scalar
    end
    if _blockaligned(_cscmat(Arb), _kdim(reterms[r]), _kdim(reterms[b]))
        return :selected
    end
    return :dense
end

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
        Lcc = _diagdata(L[kp1choose2(c)])
        X[c, c] = if isa(Lcc, Diagonal)
            Diagonal(Vector{T}(undef, size(Lcc, 1)))
        elseif isa(Lcc, UniformBlockDiagonal)
            UniformBlockDiagonal(Array{T,3}(undef, size(Lcc.data)))
        elseif isa(Lcc, TriangularRFP)
            # mirror the packed storage of the L block
            TriangularRFP(Matrix{T}(undef, size(Lcc.data)), Lcc.transr, Lcc.uplo)
        else
            Matrix{T}(undef, size(Lcc))
        end
        for r in (c + 1):nb
            X[r, c] = if r ≤ k && _sparseXok(L, X, reterms, r, c)
                _patternmirror(_cscmat(L[block(r, c)]))
            else
                Matrix{T}(undef, size(L[block(r, c)]))
            end
        end
    end
    maxheavy = 0
    path = fill(:none, nb, k)
    for b in 1:k
        Abb = A[kp1choose2(b)]
        S[b, b] = if isa(Abb, Diagonal)
            Diagonal(Vector{T}(undef, size(Abb, 1)))
        else    # UniformBlockDiagonal
            UniformBlockDiagonal(Array{T,3}(undef, size(Abb.data)))
        end
        S[nb, b] = Matrix{T}(undef, size(A[block(nb, b)]))
        for r in (b + 1):k
            Arb = A[block(r, b)]
            path[r, b] = _pairpath(Arb, X, reterms, r, b)
            if path[r, b] === :scalar
                # entries of S accumulated directly, no buffer.  When the fill block
                # L[r,r] is dense the cross term is evaluated with a BLAS-3 kernel
                # needing a q_r × GRAD_PANEL scratch (see `_crosspair_blas3!`)
                if isa(_diagdata(L[kp1choose2(r)]), Union{Matrix,TriangularRFP})
                    maxheavy = max(maxheavy, size(Arb, 1))
                end
            elseif path[r, b] === :selected
                S[r, b] = _patternmirror(_cscmat(Arb))
            else
                S[r, b] = Matrix{T}(undef, size(Arb))
                C1[r, b] = Matrix{T}(undef, size(Arb))
                C2[r, b] = Matrix{T}(undef, size(Arb))
            end
        end
    end
    G = [Matrix{T}(undef, _kdim(rt), _kdim(rt)) for rt in reterms]
    Ppanel = Matrix{T}(undef, maxheavy, iszero(maxheavy) ? 0 : GRAD_PANEL)
    return GradientWorkspace{T}(X, S, C1, C2, G, Ppanel, path)
end

#####
##### blocked computation of the lower blocks of X = L⁻¹
#####

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

function _mulsub!(C::Matrix{T}, A::Matrix{T}, B::UniformBlockDiagonal{T}) where {T}
    dat = B.data
    kc = size(dat, 1)
    for f in axes(dat, 3)
        cols = ((f - 1) * kc + 1):(f * kc)
        mul!(view(C, :, cols), view(A, :, cols), view(dat, :, :, f), -one(T), one(T))
    end
    return C
end

function _mulsub!(C::Matrix{T}, A::SparseMatrixCSC{T}, B::UniformBlockDiagonal{T}) where {T}
    dat = B.data
    kc = size(dat, 1)
    rv = rowvals(A)
    nz = nonzeros(A)
    @inbounds for f in axes(dat, 3)
        coff = (f - 1) * kc
        for jloc in 1:kc, wloc in 1:kc
            x = dat[wloc, jloc, f]
            iszero(x) && continue
            for idx in nzrange(A, coff + wloc)
                C[rv[idx], coff + jloc] -= nz[idx] * x
            end
        end
    end
    return C
end

function _mulsub!(C::Matrix{T}, A::SparseMatrixCSC{T}, B::SparseMatrixCSC{T}) where {T}
    Arv = rowvals(A)
    Anz = nonzeros(A)
    Brv = rowvals(B)
    Bnz = nonzeros(B)
    @inbounds for v in axes(B, 2)
        for bidx in nzrange(B, v)
            w = Brv[bidx]
            x = Bnz[bidx]
            for aidx in nzrange(A, w)
                C[Arv[aidx], v] -= Anz[aidx] * x
            end
        end
    end
    return C
end

# sparse X block with the same pattern as A (a mirror of the L block)
function _mulsub!(C::SparseMatrixCSC{T}, A::SparseMatrixCSC{T}, B::Diagonal{T}) where {T}
    Cnz = nonzeros(C)
    Anz = nonzeros(A)
    d = B.diag
    @inbounds for v in axes(A, 2)
        x = d[v]
        for idx in nzrange(A, v)
            Cnz[idx] -= Anz[idx] * x
        end
    end
    return C
end

function _mulsub!(
    C::SparseMatrixCSC{T}, A::SparseMatrixCSC{T}, B::UniformBlockDiagonal{T}
) where {T}
    # pattern(C) == pattern(A), columns block-aligned (checked at construction)
    dat = B.data
    kc = size(dat, 1)
    Cnz = nonzeros(C)
    Anz = nonzeros(A)
    colptr = A.colptr
    @inbounds for f in axes(dat, 3)
        v1 = (f - 1) * kc + 1
        rng = Int(colptr[v1]):(Int(colptr[v1 + kc]) - 1)
        isempty(rng) && continue
        mul!(reshape(view(Cnz, rng), :, kc), reshape(view(Anz, rng), :, kc),
            view(dat, :, :, f), -one(T), one(T))
    end
    return C
end

# sparse X block accumulating a sparse-sparse product; pattern(A * B) ⊆ pattern(C)
# is verified at workspace construction (`_productcontained`)
function _mulsub!(
    C::SparseMatrixCSC{T}, A::SparseMatrixCSC{T}, B::SparseMatrixCSC{T}
) where {T}
    Crv = rowvals(C)
    Cnz = nonzeros(C)
    Arv = rowvals(A)
    Anz = nonzeros(A)
    Brv = rowvals(B)
    Bnz = nonzeros(B)
    for v in axes(B, 2)
        crng = nzrange(C, v)
        for bidx in nzrange(B, v)
            w = Brv[bidx]
            x = Bnz[bidx]
            ci = first(crng)
            for aidx in nzrange(A, w)
                u = Arv[aidx]
                while Crv[ci] < u
                    ci += 1
                end
                Cnz[ci] -= Anz[aidx] * x
            end
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

function _ldivL!(Ljj::UniformBlockDiagonal{T}, B::UniformBlockDiagonal{T}) where {T}
    Ld = Ljj.data
    Bd = B.data
    for f in axes(Ld, 3)
        ldiv!(LowerTriangular(view(Ld, :, :, f)), view(Bd, :, :, f))
    end
    return B
end

function _ldivL!(Ljj::Diagonal{T}, B::SparseMatrixCSC{T}) where {T}
    d = Ljj.diag
    rv = rowvals(B)
    nz = nonzeros(B)
    @inbounds for idx in eachindex(nz)
        nz[idx] /= d[rv[idx]]
    end
    return B
end

function _ldivL!(Ljj::UniformBlockDiagonal{T}, B::SparseMatrixCSC{T}) where {T}
    dat = Ljj.data
    kr = size(dat, 1)
    rv = rowvals(B)
    nz = nonzeros(B)
    for v in axes(B, 2)
        rng = nzrange(B, v)
        i = Int(first(rng))
        while i ≤ last(rng)
            # complete kr-run on a block boundary (checked at construction)
            g = (Int(rv[i]) - 1) ÷ kr
            ldiv!(LowerTriangular(view(dat, :, :, g + 1)), view(nz, i:(i + kr - 1)))
            i += kr
        end
    end
    return B
end

_ldivL!(Ljj::TriangularRFP{T}, B::Matrix{T}) where {T} = ldiv!(Ljj, B)

_zero!(X::Matrix{T}) where {T} = fill!(X, zero(T))
_zero!(X::SparseMatrixCSC{T}) where {T} = (fill!(nonzeros(X), zero(T)); X)

_identity!(D::Diagonal{T}) where {T} = (fill!(D.diag, one(T)); D)
function _identity!(X::Matrix{T}) where {T}
    fill!(X, zero(T))
    @inbounds for i in diagind(X)
        X[i] = one(T)
    end
    return X
end

function _identity!(U::UniformBlockDiagonal{T}) where {T}
    dat = fill!(U.data, zero(T))
    @inbounds for f in axes(dat, 3), i in axes(dat, 1)
        dat[i, i, f] = one(T)
    end
    return U
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
        Lcc = _diagdata(L[kp1choose2(c)])
        Xcc = X[c, c]
        if isa(Xcc, TriangularRFP{T})
            # invert in the packed storage, in place (LAPACK's tftri)
            copyto!(Xcc.data, (Lcc::TriangularRFP{T}).data)
            inv!(Xcc)
        else
            _ldivL!(Lcc, _identity!(Xcc))
        end
        for r in (c + 1):nb
            Xrc = X[r, c]
            if isa(Lcc, TriangularRFP{T})
                # the blocks below an RFP diagonal block are dense and its packed
                # inverse cannot be multiplied against directly, so the s = c term
                # -L[r,c] X[c,c] is evaluated as a triangular solve against L[c,c]
                # in the packed storage (LAPACK's tfsm)
                Xrc = Xrc::Matrix{T}
                copyto!(Xrc, L[block(r, c)]::Matrix{T})
                rdiv!(Xrc, Lcc)
                rmul!(Xrc, -one(T))
                for s in (c + 1):(r - 1)
                    _mulsub!(Xrc, L[block(r, s)], X[s, c])
                end
            else
                _zero!(Xrc)
                for s in c:(r - 1)
                    _mulsub!(Xrc, L[block(r, s)], X[s, c])
                end
            end
            _ldivL!(_diagdata(L[kp1choose2(r)]), Xrc)
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

# accumulate S += Xsrᵀ Xsb for the block types that occur in X
function _gramacc!(S::Matrix{T}, Xsr::AbstractMatrix{T}, Xsb::AbstractMatrix{T}) where {T}
    return mul!(S, Xsr', Xsb, one(T), one(T))
end

function _gramacc!(S::Matrix{T}, Xsr::UniformBlockDiagonal{T}, Xsb::Matrix{T}) where {T}
    dat = Xsr.data
    kr = size(dat, 1)
    for f in axes(dat, 3)
        rows = ((f - 1) * kr + 1):(f * kr)
        mul!(view(S, rows, :), adjoint(view(dat, :, :, f)), view(Xsb, rows, :),
            one(T), one(T))
    end
    return S
end

function _gramacc!(
    S::Matrix{T}, Xsr::UniformBlockDiagonal{T}, Xsb::SparseMatrixCSC{T}
) where {T}
    dat = Xsr.data
    kr = size(dat, 1)
    rv = rowvals(Xsb)
    nz = nonzeros(Xsb)
    for v in axes(Xsb, 2)
        rng = nzrange(Xsb, v)
        i = Int(first(rng))
        while i ≤ last(rng)
            g = (Int(rv[i]) - 1) ÷ kr     # complete kr-run on a block boundary
            mul!(view(S, (g * kr + 1):((g + 1) * kr), v),
                adjoint(view(dat, :, :, g + 1)), view(nz, i:(i + kr - 1)),
                one(T), one(T))
            i += kr
        end
    end
    return S
end

# S += Xsr' Xsb for a packed triangular Xsr = [X11 0; X21 X22]: two triangular
# multiplications (BLAS's trmm on the storage regions) and one dense product
function _gramacc!(S::Matrix{T}, Xsr::TriangularRFP{T}, Xsb::Matrix{T}) where {T}
    tv, uv, m1, m2 = _rfpviews(Xsr)
    n = size(Xsr, 1)
    q = size(Xsb, 2)
    B2 = view(Xsb, (m1 + 1):n, :)
    tmp = Matrix{T}(undef, m1, q)
    copyto!(tmp, view(Xsb, 1:m1, :))
    BLAS.trmm!('L', 'L', 'T', 'N', one(T), view(tv, 1:m1, :), tmp)  # X11' B1
    view(S, 1:m1, :) .+= tmp
    mul!(view(S, 1:m1, :), adjoint(view(tv, (m1 + 1):n, :)), B2, one(T), one(T))
    if !iszero(m2)
        tmp2 = view(tmp, 1:m2, :)
        copyto!(tmp2, B2)
        BLAS.trmm!('L', 'U', 'N', 'N', one(T), uv, tmp2)            # X22' B2
        view(S, (m1 + 1):n, :) .+= tmp2
    end
    return S
end

function _gramacc!(S::Matrix{T}, Xsr::Diagonal{T}, Xsb::SparseMatrixCSC{T}) where {T}
    d = Xsr.diag
    rv = rowvals(Xsb)
    nz = nonzeros(Xsb)
    @inbounds for v in axes(Xsb, 2)
        for idx in nzrange(Xsb, v)
            u = rv[idx]
            S[u, v] += d[u] * nz[idx]
        end
    end
    return S
end

# dense S block for the pair (r, b), r > b or r == nb (the [Xy] row)
function _gram!(S::Matrix{T}, w::GradientWorkspace{T}, r::Int, b::Int, kre::Int,
    wx::T, wy::T) where {T}
    X = w.X
    fill!(S, zero(T))
    for s in r:kre
        _gramacc!(S, X[s, r], X[s, b])
    end
    return _xycorrection!(S, X[kre + 1, r]::Matrix{T}, X[kre + 1, b]::Matrix{T}, wx, wy)
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

function _colsumabs2!(d::Vector{T}, X::SparseMatrixCSC{T}) where {T}
    nz = nonzeros(X)
    @inbounds for v in axes(X, 2)
        acc = zero(T)
        for idx in nzrange(X, v)
            acc += abs2(nz[idx])
        end
        d[v] += acc
    end
    return d
end

function _colsumabs2!(d::Vector{T}, X::TriangularRFP{T}) where {T}
    tv, uv, m1, m2 = _rfpviews(X)
    n = length(d)
    @inbounds for j in 1:m1
        d[j] += sum(abs2, view(tv, j:n, j))
    end
    @inbounds for jl in 1:m2
        d[m1 + jl] += sum(abs2, view(uv, jl, jl:m2))
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

# face-diagonal accumulation dat[:,:,f] += X[:,face f]ᵀ X[:,face f] for the block
# types that occur in X (used when A[b,b] is UniformBlockDiagonal)
function _gramfaces_acc!(dat::Array{T,3}, Xsb::Matrix{T}) where {T}
    kb = size(dat, 1)
    for f in axes(dat, 3)
        cols = ((f - 1) * kb + 1):(f * kb)
        Xv = view(Xsb, :, cols)
        mul!(view(dat, :, :, f), Xv', Xv, one(T), one(T))
    end
    return dat
end

function _gramfaces_acc!(dat::Array{T,3}, Xsb::UniformBlockDiagonal{T}) where {T}
    Xd = Xsb.data
    for f in axes(dat, 3)
        Xf = view(Xd, :, :, f)
        mul!(view(dat, :, :, f), Xf', Xf, one(T), one(T))
    end
    return dat
end

function _gramfaces_acc!(dat::Array{T,3}, Xsb::TriangularRFP{T}) where {T}
    kb = size(dat, 1)
    for f in axes(dat, 3)
        coff = (f - 1) * kb
        for c in 1:kb, a in 1:kb
            dat[a, c, f] += _xdotRFP(Xsb, coff + a, coff + c)
        end
    end
    return dat
end

function _gramfaces_acc!(dat::Array{T,3}, Xsb::SparseMatrixCSC{T}) where {T}
    kb = size(dat, 1)
    nzv = nonzeros(Xsb)
    colptr = Xsb.colptr
    for f in axes(dat, 3)
        v1 = (f - 1) * kb + 1
        rng = Int(colptr[v1]):(Int(colptr[v1 + kb]) - 1)
        isempty(rng) && continue
        # the kb columns of the block share their row pattern (checked at
        # construction), so the reshaped nonzeros are the dense column slices
        M = reshape(view(nzv, rng), :, kb)
        mul!(view(dat, :, :, f), M', M, one(T), one(T))
    end
    return dat
end

function _gramfaces!(S::UniformBlockDiagonal{T}, w::GradientWorkspace{T}, b::Int,
    kre::Int, wx::T, wy::T) where {T}
    dat = fill!(S.data, zero(T))
    kb = size(dat, 1)
    for s in b:kre
        _gramfaces_acc!(dat, w.X[s, b])
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

# entry (u, v) of Xrᵀ Xb for the dense-Xb block types of X, shared by the
# scalar path (`_sparseacc`) and the selected-entry path (`_selacc!`)
function _xdot(Xr::Matrix{T}, Xb::Matrix{T}, u::Integer, v::Integer) where {T}
    return dot(view(Xr, :, u), view(Xb, :, v))
end

function _xdot(Xr::Diagonal{T}, Xb::Matrix{T}, u::Integer, v::Integer) where {T}
    return Xr.diag[u] * Xb[u, v]
end

function _xdot(Xr::UniformBlockDiagonal{T}, Xb::Matrix{T}, u::Integer, v::Integer) where {T}
    dat = Xr.data
    kr = size(dat, 1)
    g, ul = divrem(Int(u) - 1, kr)
    roff = g * kr
    acc = zero(T)
    @inbounds for i in 1:kr
        acc += dat[i, ul + 1, g + 1] * Xb[roff + i, v]
    end
    return acc
end

function _xdot(Xr::TriangularRFP{T}, Xb::Matrix{T}, u::Integer, v::Integer) where {T}
    tv, uv, m1, m2 = _rfpviews(Xr)
    n = size(Xr, 1)
    ui = Int(u)
    if ui ≤ m1
        return dot(view(tv, ui:n, ui), view(Xb, ui:n, v))
    else
        ul = ui - m1
        return dot(view(uv, ul, ul:m2), view(Xb, ui:n, v))
    end
end

# entry (a, b) of X'X for an RFP-stored lower-triangular X; only rows ≥ max(a, b)
# contribute, which selects the storage regions of the two columns
function _xdotRFP(X::TriangularRFP{T}, a::Integer, b::Integer) where {T}
    tv, uv, m1, m2 = _rfpviews(X)
    n = size(X, 1)
    lo, hi = minmax(Int(a), Int(b))
    if hi ≤ m1
        return dot(view(tv, hi:n, lo), view(tv, hi:n, hi))
    elseif lo > m1
        ll = lo - m1
        hl = hi - m1
        return dot(view(uv, ll, hl:m2), view(uv, hl, hl:m2))
    else
        hl = hi - m1
        return dot(view(tv, hi:n, lo), view(uv, hl, hl:m2))
    end
end

# entry (u, v) of Xkrᵀ W Xkb for the [Xy] block row, with weight wx on the
# first p rows and wy on the last
function _xydot(Xkr::Matrix{T}, Xkb::Matrix{T}, u::Integer, v::Integer,
    wx::T, wy::T) where {T}
    plast = size(Xkr, 1)
    s = wy * Xkr[plast, u] * Xkb[plast, v]
    if !iszero(wx)
        @inbounds for i in 1:(plast - 1)
            s += wx * Xkr[i, u] * Xkb[i, v]
        end
    end
    return s
end

# Σ over the nonzeros (u, v) of A of A[u,v] * (Xrᵀ Xb)[u,v] for one block-row pair of X.
# This method is a function barrier: the X blocks are stored with an abstract element
# type and the entry loops must run with concretely typed arrays.
function _sparseacc(A::SparseMatrixCSC{T},
    Xr::Union{Diagonal{T},Matrix{T},TriangularRFP{T}},
    Xb::Matrix{T}) where {T}
    rv = rowvals(A)
    nz = nonzeros(A)
    acc = zero(T)
    @inbounds for v in axes(A, 2)
        for idx in nzrange(A, v)
            acc += nz[idx] * _xdot(Xr, Xb, rv[idx], v)
        end
    end
    return acc
end

# ditto for the [Xy] block row
function _sparseaccxy(A::SparseMatrixCSC{T}, Xkr::Matrix{T}, Xkb::Matrix{T},
    wx::T, wy::T) where {T}
    rv = rowvals(A)
    nz = nonzeros(A)
    acc = zero(T)
    @inbounds for v in axes(A, 2)
        for idx in nzrange(A, v)
            acc += nz[idx] * _xydot(Xkr, Xkb, rv[idx], v, wx, wy)
        end
    end
    return acc
end

#####
##### selected entries of S on the pattern of a sparse A block (any term dimensions)
#####

# accumulate Sp[u,v] += (Xsrᵀ Xsb)[u,v] over the nonzeros (u,v) of Sp for the block
# types that occur in X; like `_sparseacc` these are function barriers
function _selacc!(Sp::SparseMatrixCSC{T},
    Xr::Union{Diagonal{T},UniformBlockDiagonal{T},Matrix{T},TriangularRFP{T}},
    Xb::Matrix{T}) where {T}
    rv = rowvals(Sp)
    nz = nonzeros(Sp)
    @inbounds for v in axes(Sp, 2)
        for idx in nzrange(Sp, v)
            nz[idx] += _xdot(Xr, Xb, rv[idx], v)
        end
    end
    return Sp
end

function _selacc!(Sp::SparseMatrixCSC{T}, Xr::Diagonal{T}, Xb::SparseMatrixCSC{T}) where {T}
    d = Xr.diag
    rv = rowvals(Sp)
    nz = nonzeros(Sp)
    Brv = rowvals(Xb)
    Bnz = nonzeros(Xb)
    @inbounds for v in axes(Sp, 2)
        brng = nzrange(Xb, v)
        bi = Int(first(brng))
        for idx in nzrange(Sp, v)
            u = rv[idx]
            while bi ≤ last(brng) && Brv[bi] < u
                bi += 1
            end
            bi ≤ last(brng) || break
            if Brv[bi] == u
                nz[idx] += d[u] * Bnz[bi]
            end
        end
    end
    return Sp
end

function _selacc!(
    Sp::SparseMatrixCSC{T}, Xr::UniformBlockDiagonal{T}, Xb::SparseMatrixCSC{T}
) where {T}
    dat = Xr.data
    kr = size(dat, 1)
    rv = rowvals(Sp)
    nz = nonzeros(Sp)
    Brv = rowvals(Xb)
    Bnz = nonzeros(Xb)
    @inbounds for v in axes(Sp, 2)
        brng = nzrange(Xb, v)
        bi = Int(first(brng))
        for idx in nzrange(Sp, v)
            g, ul = divrem(Int(rv[idx]) - 1, kr)
            rowstart = g * kr + 1
            while bi ≤ last(brng) && Brv[bi] < rowstart
                bi += 1
            end
            # Xb is block-aligned: a face either contributes a complete kr-run or
            # nothing at all to this column
            if bi ≤ last(brng) && Brv[bi] == rowstart
                acc = zero(T)
                for i in 1:kr
                    acc += dat[i, ul + 1, g + 1] * Bnz[bi + i - 1]
                end
                nz[idx] += acc
            end
        end
    end
    return Sp
end

function _selacc!(Sp::SparseMatrixCSC{T}, Xr::Matrix{T}, Xb::SparseMatrixCSC{T}) where {T}
    rv = rowvals(Sp)
    nz = nonzeros(Sp)
    Brv = rowvals(Xb)
    Bnz = nonzeros(Xb)
    @inbounds for v in axes(Sp, 2)
        brng = nzrange(Xb, v)
        isempty(brng) && continue
        for idx in nzrange(Sp, v)
            u = rv[idx]
            acc = zero(T)
            for bi in brng
                acc += Xr[Brv[bi], u] * Bnz[bi]
            end
            nz[idx] += acc
        end
    end
    return Sp
end

function _selacc!(Sp::SparseMatrixCSC{T}, Xr::SparseMatrixCSC{T}, Xb::Matrix{T}) where {T}
    rv = rowvals(Sp)
    nz = nonzeros(Sp)
    Rrv = rowvals(Xr)
    Rnz = nonzeros(Xr)
    @inbounds for v in axes(Sp, 2)
        for idx in nzrange(Sp, v)
            acc = zero(T)
            for ri in nzrange(Xr, rv[idx])
                acc += Rnz[ri] * Xb[Rrv[ri], v]
            end
            nz[idx] += acc
        end
    end
    return Sp
end

function _selacc!(
    Sp::SparseMatrixCSC{T}, Xr::SparseMatrixCSC{T}, Xb::SparseMatrixCSC{T}
) where {T}
    rv = rowvals(Sp)
    nz = nonzeros(Sp)
    Rrv = rowvals(Xr)
    Rnz = nonzeros(Xr)
    Brv = rowvals(Xb)
    Bnz = nonzeros(Xb)
    @inbounds for v in axes(Sp, 2)
        brng = nzrange(Xb, v)
        isempty(brng) && continue
        for idx in nzrange(Sp, v)
            rrng = nzrange(Xr, rv[idx])
            acc = zero(T)
            ri = Int(first(rrng))
            bi = Int(first(brng))
            while ri ≤ last(rrng) && bi ≤ last(brng)
                rw = Rrv[ri]
                bw = Brv[bi]
                if rw == bw
                    acc += Rnz[ri] * Bnz[bi]
                    ri += 1
                    bi += 1
                elseif rw < bw
                    ri += 1
                else
                    bi += 1
                end
            end
            nz[idx] += acc
        end
    end
    return Sp
end

# the weighted [Xy]-block-row correction on the pattern of Sp
function _selaccxy!(Sp::SparseMatrixCSC{T}, Xkr::Matrix{T}, Xkb::Matrix{T},
    wx::T, wy::T) where {T}
    rv = rowvals(Sp)
    nz = nonzeros(Sp)
    @inbounds for v in axes(Sp, 2)
        for idx in nzrange(Sp, v)
            nz[idx] += _xydot(Xkr, Xkb, rv[idx], v, wx, wy)
        end
    end
    return Sp
end

# contract the selected entries of S against the nonzero k_r × k_b blocks of A:
#   G_b += (Λᵣᵀ A_blk)ᵀ S_blk = A_blkᵀ (Λᵣ S_blk),   G_r += (A_blk Λ_b) S_blkᵀ
function _selcontract!(Gb::Matrix{T}, Gr::Matrix{T}, A::SparseMatrixCSC{T},
    Sp::SparseMatrixCSC{T}, rtr::ReMat{T}, rtb::ReMat{T}) where {T}
    kr = _kdim(rtr)
    kb = _kdim(rtb)
    λr = rtr.λ
    λb = rtb.λ
    Anz = nonzeros(A)
    Snz = nonzeros(Sp)
    Ablk = Matrix{T}(undef, kr, kb)
    Sblk = Matrix{T}(undef, kr, kb)
    t = Matrix{T}(undef, kr, kb)
    @inbounds for jblk in 1:(size(A, 2) ÷ kb)
        v1 = (jblk - 1) * kb + 1
        nnzcol = length(nzrange(A, v1))
        for off in 0:kr:(nnzcol - 1)
            for j in 1:kb
                rngj = nzrange(A, v1 + j - 1)
                for i in 1:kr
                    idx = rngj[off + i]
                    Ablk[i, j] = Anz[idx]
                    Sblk[i, j] = Snz[idx]
                end
            end
            mul!(t, λr, Sblk)
            mul!(Gb, Ablk', t, one(T), one(T))
            mul!(t, Ablk, λb)
            mul!(Gr, t, Sblk', one(T), one(T))
        end
    end
    return nothing
end

# off-diagonal pair (r, b) with sparse A[r,b] between terms of any dimension: the
# entries of S on the sparsity pattern of A are accumulated into the sparse buffer
# S[r,b] (sharing A's pattern) and contracted blockwise against A
function _selectedpair!(w::GradientWorkspace{T}, m::LinearMixedModel{T}, r::Int, b::Int,
    wx::T, wy::T) where {T}
    (; A, reterms) = m
    kre = length(reterms)
    Sp = w.S[r, b]::SparseMatrixCSC{T}
    fill!(nonzeros(Sp), zero(T))
    for s in r:kre
        _selacc!(Sp, w.X[s, r], w.X[s, b])
    end
    _selaccxy!(Sp, w.X[kre + 1, r]::Matrix{T}, w.X[kre + 1, b]::Matrix{T}, wx, wy)
    _selcontract!(w.G[b], w.G[r], _cscmat(A[block(r, b)]), Sp, reterms[r], reterms[b])
    return w
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

# off-diagonal pair (r, b) between two scalar terms with sparse A[r,b] and dense X
# blocks: only the entries of S on the sparsity pattern of A are evaluated
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

# ditto for a packed triangular fill block Xrr = [X11 0; X21 X22]: the panel product
# is assembled from two triangular multiplications (BLAS's trmm on the storage
# regions) and one dense product
function _crossacc_blas3!(
    Pp::Matrix{T}, A::SparseMatrixCSC{T}, Xrr::TriangularRFP{T}, Xrb::Matrix{T}
) where {T}
    rv = rowvals(A)
    nz = nonzeros(A)
    tv, uv, m1, m2 = _rfpviews(Xrr)
    qr = size(Xrr, 2)
    qb = size(Xrb, 2)
    acc = zero(T)
    coloff = 0
    while coloff < qb
        width = min(size(Pp, 2), qb - coloff)
        cols = (coloff + 1):(coloff + width)
        P1 = view(Pp, 1:m1, 1:width)
        copyto!(P1, view(Xrb, 1:m1, cols))
        BLAS.trmm!('L', 'L', 'T', 'N', one(T), view(tv, 1:m1, :), P1)  # X11' Y1
        mul!(P1, adjoint(view(tv, (m1 + 1):qr, :)), view(Xrb, (m1 + 1):qr, cols),
            one(T), one(T))                                            # += X21' Y2
        if !iszero(m2)
            P2 = view(Pp, (m1 + 1):qr, 1:width)
            copyto!(P2, view(Xrb, (m1 + 1):qr, cols))
            BLAS.trmm!('L', 'U', 'N', 'N', one(T), uv, P2)             # X22' Y2
        end
        Pv = view(Pp, 1:qr, 1:width)
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
    isa(w.X[r, r], Union{Matrix,TriangularRFP}) || return false
    Arb = _cscmat(m.A[block(r, b)])
    return nnz(Arb) > 0.03 * size(Arb, 1) * size(Arb, 2)
end

# same result as `_sparsepair!`, but the heavy s = r term uses the BLAS-3 kernel
function _crosspair_blas3!(w::GradientWorkspace{T}, m::LinearMixedModel{T}, r::Int, b::Int,
    wx::T, wy::T) where {T}
    (; A, reterms) = m
    kre = length(reterms)
    Arb = _cscmat(A[block(r, b)])
    acc = _crossacc_blas3!(w.Ppanel, Arb, w.X[r, r], w.X[r, b]::Matrix{T})
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
especially for models with many covariance parameters.  The blocks of `L⁻¹` mirror the
block-diagonal and sparse (nested-grouping) structure of `L`, so the workspace storage is
of the same order as `L` itself; models with two large *crossed* (non-nested) grouping
factors store dense off-diagonal blocks, for which the storage can be substantial.
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
            path = w.path[r, b]
            if path === :selected
                _selectedpair!(w, m, r, b, wx, wy)
            elseif path === :scalar
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
