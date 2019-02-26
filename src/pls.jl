"""
    densify(S::SparseMatrix, threshold=0.3)

Convert sparse `S` to `Diagonal` if `S` is diagonal or to `full(S)` if
the proportion of nonzeros exceeds `threshold`.
"""
function densify(A::SparseMatrixCSC, threshold::Real = 0.3)
    S = sparse(A)
    m, n = size(S)
    if m == n && isdiag(S)  # convert diagonal sparse to Diagonal
        Diagonal(diag(S))
    elseif nnz(S)/(S.m * S.n) ≤ threshold
        A
    else
        Array(S)
    end
end
densify(A::AbstractMatrix, threshold::Real = 0.3) = A
function densify(A::BlockedSparse, threshold::Real=0.3)
    Asp = A.cscmat
    Ad = densify(Asp)
    Ad === Asp ? A : Ad
end

function LinearMixedModel(f, trms, wts)
    n = size(trms[1], 1)
    T = eltype(trms[end])
    sqrtwts = sqrt.(wts)
    sz = size.(trms, 2)
    if !isempty(wts)
        reweight!.(trms, Vector[sqrtwts])
    end
    nt = length(trms)
    A = BlockArrays._BlockArray(AbstractMatrix{T}, sz, sz)
    L = BlockArrays._BlockArray(AbstractMatrix{T}, sz, sz)
    for j in 1:nt, i in j:nt
        Lij = L[Block(i,j)] = densify(trms[i]'trms[j])
        A[Block(i,j)] = deepcopy(isa(Lij, BlockedSparse) ? Lij.cscmat : Lij)
    end
                  # check for fill-in due to non-nested grouping factors
    for i in 2:nt
        ti = trms[i]
        if isa(ti, AbstractFactorReTerm)
            for j in 1:(i - 1)
                tj = trms[j]
                if isa(tj, AbstractFactorReTerm) && !isnested(tj, ti)
                    for k in i:nt
                        L[Block(k, i)] = Matrix(L[Block(k, i)])
                    end
                    break
                end
            end
        end
    end
    optsum = OptSummary(getθ(trms), lowerbd(trms), :LN_BOBYQA;
        ftol_rel = T(1.0e-12), ftol_abs = T(1.0e-8))
    fill!(optsum.xtol_abs, 1.0e-10)
    LinearMixedModel(f, trms, sqrtwts, A, LowerTriangular(L), optsum)
end

function LinearMixedModel(f::Formula, fr::AbstractDataFrame;
    weights::Vector = [], contrasts=Dict(), rdist::Distribution=Normal())
    mf = ModelFrame(f, fr, contrasts=contrasts)
    X = ModelMatrix(mf).m
    n = size(X, 1)
    T = eltype(X)
    y = model_response(mf, rdist)
    tdict = Dict{Symbol, Vector{Any}}()
    for t in filter(x -> Meta.isexpr(x, :call) && x.args[1] == :|, mf.terms.terms)
        fnm = t.args[3]
        isa(fnm, Symbol) || throw(ArgumentError("rhs of $t must be a symbol"))
        tdict[fnm] = haskey(tdict, fnm) ? push!(tdict[fnm], t.args[2]) : [t.args[2]]
    end
    isempty(tdict) && throw(ArgumentError("No random-effects terms found in $f"))
    trms = AbstractTerm{T}[]
    for (grp, lhs) in tdict
        gr = compress(categorical(getindex(mf.df, grp)))
        if (length(lhs) == 1 && lhs[1] == 1)
            push!(trms, ScalarFactorReTerm(gr, grp))
        else
            m = T[]
            coefnms = String[]
            trsize = Int[]
            for l in lhs
                if l == 1
                    append!(m, ones(T, n))
                    push!(coefnms, "(Intercept)")
                    push!(trsize, 1)
                else
                    fr = ModelFrame(@eval(@formula($nothing ~ $l)), mf.df)
                    append!(m, ModelMatrix(fr).m)
                    cnms = coefnames(fr)
                    append!(coefnms, cnms)
                    push!(trsize, length(cnms))
                end
            end
            push!(trms,
                  length(coefnms) == 1 ? ScalarFactorReTerm(gr, m, grp, coefnms) :
                  VectorFactorReTerm(gr, collect(adjoint(reshape(m, (n, sum(trsize))))), grp,
                      coefnms,  trsize))
        end
    end
    sort!(trms, by = nrandomeff, rev = true)
    push!(trms, MatrixTerm(X, coefnames(mf)))
    push!(trms, MatrixTerm(y))
    LinearMixedModel(f, trms, oftype(y, weights))
end

StatsBase.model_response(mf::ModelFrame, d::Distribution) =
    model_response(mf.df[mf.terms.eterms[1]], d)

StatsBase.model_response(v::AbstractVector, d::Distribution) = Vector{partype(d)}(v)

function StatsBase.model_response(v::CategoricalVector, d::Bernoulli)
    levs = levels(v)
    nlevs = length(levs)
    @argcheck(nlevs ≤ 2)
    nlevs < 2 ? zeros(v, partype(d)) : partype(d)[cv == levs[2] for cv in v]
end

StatsBase.fit(::Type{LinearMixedModel},
              f::Formula,
              fr::AbstractDataFrame;
              weights=[],
              contrasts=Dict(),
              verbose=false,
              REML=false) =
    fit!(LinearMixedModel(f, fr, weights=weights, contrasts=contrasts),
         verbose=verbose,
         REML=REML)

"""
    updateL!(m::LinearMixedModel)

Update the blocked lower Cholesky factor, `m.L`, from `m.A` and `m.trms` (used for Λ only)

This is the crucial step in evaluating the objective, given a new parameter value.
"""
function updateL!(m::LinearMixedModel{T}) where {T}
    A = m.A
    Ldat = m.L.data
    trms = m.trms
    nblk = length(trms)
    for (j, trm) in enumerate(trms)
        Ljj = scaleInflate!(Ldat[Block(j, j)], A[Block(j, j)], trm)
        LjjH = isa(Ljj, Diagonal) ? Ljj : Hermitian(Ljj, :L)
        LjjT = isa(Ljj, Diagonal) ? Ljj : LowerTriangular(Ljj)
        for jj in 1:(j - 1)
            rankUpdate!(LjjH, Ldat[Block(j, jj)], -one(T))
        end
        cholUnblocked!(Ljj, Val{:L})
        for i in (j + 1):nblk
            Lij = lmul!(Λ(trms[i])', rmul!(copyto!(Ldat[Block(i, j)], A[Block(i, j)]), Λ(trm)))
            for jj in 1:(j - 1)
                mulαβ!(Lij, Ldat[Block(i, jj)], Ldat[Block(j, jj)]', -one(T), one(T))
            end
            rdiv!(Lij, LjjT')
        end
    end
    m
end

StatsBase.coef(m::MixedModel) = fixef(m, false)

"""
    fit!(m::LinearMixedModel; verbose=false, REML=false)

Optimize the objective of a `LinearMixedModel`.  When `verbose` is `true` the values of the
objective and the parameters are printed on stdout at each function evaluation.  The
objective is negative twice the log-likelihood when `REML` is `false` (the default) or
the REML criterion.
"""
function StatsBase.fit!(m::LinearMixedModel{T}; verbose=false, REML=nothing) where {T}
    optsum = m.optsum
    opt = Opt(optsum)
    feval = 0
    if isa(REML, Bool)
        optsum.REML = REML
    end
    function obj(x, g)
        isempty(g) || error("gradient not defined")
        feval += 1
        val = objective(updateL!(setθ!(m, x)))
        feval == 1 && (optsum.finitial = val)
        verbose && println("f_", feval, ": ", round(val, digits=5), " ", x)
        val
    end
    NLopt.min_objective!(opt, obj)
    fmin, xmin, ret = NLopt.optimize!(opt, copyto!(optsum.final, optsum.initial))
    ## check if small non-negative parameter values can be set to zero
    xmin_ = copy(xmin)
    lb = optsum.lowerbd
    for i in eachindex(xmin_)
        if iszero(lb[i]) && zero(T) < xmin_[i] < T(0.001)
            xmin_[i] = zero(T)
        end
    end
    if xmin_ ≠ xmin
        if (zeroobj = obj(xmin_, T[])) ≤ (fmin + 1.e-5)
            fmin = zeroobj
            copyto!(xmin, xmin_)
        end
    end
    ## ensure that the parameter values saved in m are xmin
    updateL!(setθ!(m, xmin))

    optsum.feval = feval
    optsum.final = xmin
    optsum.fmin = fmin
    optsum.returnvalue = ret
    ret == :ROUNDOFF_LIMITED && @warn("NLopt was roundoff limited")
    if ret ∈ [:FAILURE, :INVALID_ARGS, :OUT_OF_MEMORY, :FORCED_STOP, :MAXFEVAL_REACHED]
        @warn("NLopt optimization failure: $ret")
    end
    m
end

function fitted!(v::AbstractArray{T}, m::LinearMixedModel{T}) where {T}
    ## FIXME: Create and use `effects(m) -> β, b` w/o calculating β twice
    trms = m.trms
    mul!(vec(v), trms[end - 1], fixef(m))
    b = ranef(m)
    for j in eachindex(b)
        unscaledre!(vec(v), trms[j], b[j])
    end
    v
end

StatsBase.fitted(m::LinearMixedModel{T}) where {T} = fitted!(Vector{T}(undef, nobs(m)), m)

StatsBase.predict(m::LinearMixedModel) = fitted(m)

StatsBase.residuals(m::LinearMixedModel) = model_response(m) .- fitted(m)

"""
    lowerbd(m::LinearMixedModel)

Return the vector of lower bounds on the covariance parameter vector `θ`
"""
lowerbd(m::LinearMixedModel) = lowerbd(m.trms)

StatsBase.model_response(m::LinearMixedModel) = vec(m.trms[end].x)

"""
    objective(m::LinearMixedModel)

Return negative twice the log-likelihood of model `m` or the REML criterion,
according to the value of `m.optsum.REML`
"""
function objective(m::LinearMixedModel)
    wts = m.sqrtwts
    logdet(m) + varest_denom(m)*(1 + log2π + log(varest(m))) -
        (isempty(wts) ? 0 : 2sum(log, wts))
end

"""
    fixef!(v::Vector{T}, m::LinearMixedModel{T})

Overwrite `v` with the pivoted and, possibly, truncated fixed-effects coefficients of model `m`
"""
function fixef!(v::AbstractVector{T}, m::LinearMixedModel{T}) where {T}
    L = feL(m)
    @argcheck(length(v) == size(L, 1), DimensionMismatch)
    ldiv!(adjoint(L), copyto!(v, m.L.data.blocks[end, end - 1]))
end

"""
    fixef(m::MixedModel, permuted=true)

Return the fixed-effects parameter vector estimate of `m`.

If `permuted` is `true` the vector elements are permuted according to
`m.trms[end - 1].piv` and truncated to the rank of that term.
"""
function fixef(m::LinearMixedModel{T}, permuted=true) where {T}
    permuted && return fixef!(Vector{T}(undef, size(m)[2]), m)
    Xtrm = m.trms[end - 1]
    piv = Xtrm.piv
    v = fill(-zero(T), size(piv))
    fixef!(view(v, 1:Xtrm.rank), m)
    invpermute!(v, piv)
end

StatsBase.dof(m::LinearMixedModel) = size(m)[2] + sum(nθ, m.trms) + 1

function StatsBase.dof_residual(m::LinearMixedModel)
    (n, p, q, k) = size(m)
    n - m.optsum.REML * p
end

StatsBase.loglikelihood(m::LinearMixedModel) = -objective(m)/2

StatsBase.nobs(m::LinearMixedModel) = length(m.trms[end].wtx)

function Base.size(m::LinearMixedModel)
    trms = m.trms
    k = length(trms) - 2
    q = sum(size(trms[j], 2) for j in 1:k)
    length(trms[end]), trms[k+1].rank, q, k
end

"""
    sdest(m::LinearMixedModel)

Return the estimate of σ, the standard deviation of the per-observation noise.
"""
sdest(m::LinearMixedModel) = √varest(m)

"""
    setθ!{T}(m::LinearMixedModel{T}, v::Vector{T})

Install `v` as the θ parameters in `m`.
"""
function setθ!(m::LinearMixedModel, v)
    offset = 0
    for trm in m.trms
        if isa(trm, ScalarFactorReTerm)
            offset += 1
            setθ!(trm, v[offset])
        else
            k = nθ(trm)
            iszero(k) || setθ!(trm, view(v, (1:k) .+ offset))
            offset += k
        end
    end
    m
end

"""
    sqrtpwrss(m::LinearMixedModel)

Return the square root of the penalized, weighted residual sum-of-squares (pwrss).

This value is the contents of the `1 × 1` bottom right block of `m.L`
"""
sqrtpwrss(m::LinearMixedModel) = @views m.L.data.blocks[end, end][1]

function varest_denom(m::LinearMixedModel)
    (n, p, q, k) = size(m)
    n - m.optsum.REML * p
end

"""
    varest(m::LinearMixedModel)

Returns the estimate of σ², the variance of the conditional distribution of Y given B.
"""
varest(m::LinearMixedModel) = pwrss(m) / varest_denom(m)

"""
    pwrss(m::LinearMixedModel)

The penalized residual sum-of-squares.
"""
pwrss(m::LinearMixedModel) = abs2(sqrtpwrss(m))

function StatsBase.deviance(m::LinearMixedModel)
    isfit(m) || error("Model has not been fit")
    objective(m)
end

"""
    isfit(m::LinearMixedModel)

Return a `Bool` indicating if the model has been fit.
"""
isfit(m::LinearMixedModel) = m.optsum.fmin < Inf

"""
    lrt(mods::LinearMixedModel...)

Perform sequential likelihood ratio tests on a sequence of models.

The returned value is a `DataFrame` containing information on the likelihood ratio tests.
"""
function lrt(mods::LinearMixedModel...) # not tested
    (nm = length(mods)) > 1 || throw(ArgumentError("at least two models are required for a likelihood ratio test"))
    m1 = mods[1]
    n = nobs(m1)
    for i in 2:nm
        nobs(mods[i]) == n || throw(ArgumentError("number of observations must be constant across models"))
    end
    mods = mods[sortperm([dof(m)::Int for m in mods])]
    degf = Int[dof(m) for m in mods]
    dev = [deviance(m)::Float64 for m in mods]
    csqr = pushfirst!([(dev[i-1]-dev[i])::Float64 for i in 2:nm],NaN)
    pval = pushfirst!([ccdf(Chisq(degf[i]-degf[i-1]),csqr[i])::Float64 for i in 2:nm],NaN)
    DataFrame(Df = degf, Deviance = dev, Chisq=csqr,pval=pval)
end

"""
    reweight!{T}(m::LinearMixedModel{T}, wts::Vector{T})

Update `m.sqrtwts` from `wts` and reweight each term.  Recompute `m.A` and `m.L`.
"""
function reweight!(m::LinearMixedModel, weights)
    trms = m.trms
    m.sqrtwts .= sqrt.(weights)
    reweight!.(trms, Vector[m.sqrtwts])
    ntrm = length(trms)
    A = m.A
    for j in 1:ntrm, i in j:ntrm
        mul!(A[Block(i, j)], trms[i]', trms[j])
    end
    updateL!(m)
end

function Base.show(io::IO, m::LinearMixedModel)
    if !isfit(m)
        @warn("Model has not been fit")
        return nothing
    end
    n, p, q, k = size(m)
    REML = m.optsum.REML
    println(io, "Linear mixed model fit by ", REML ? "REML" : "maximum likelihood")
    println(io, " ", m.formula)
    oo = objective(m)
    if REML
        println(io, " REML criterion at convergence: ", oo)
    else
        nums = showoff([-oo/ 2, oo, aic(m), bic(m)])
        fieldwd = max(maximum(textwidth.(nums)) + 1, 11)
        for label in [" logLik", "-2 logLik", "AIC", "BIC"]
            print(io, rpad(lpad(label, (fieldwd + textwidth(label)) >> 1), fieldwd))
        end
        println(io)
        for num in nums
            print(io, lpad(num, fieldwd))
        end
        println(io)
    end
    println(io)

    show(io,VarCorr(m))

    gl = grplevels(m)
    @printf(io," Number of obs: %d; levels of grouping factors: %d", n, gl[1])
    for l in gl[2:end] @printf(io, ", %d", l) end
    println(io)
    @printf(io,"\n  Fixed-effects parameters:\n")
    show(io,coeftable(m))
end
