"""
    densify(S::SparseMatrix, threshold=0.3)

Convert sparse `S` to `Diagonal` if `S` is diagonal or to `full(S)` if
the proportion of nonzeros exceeds `threshold`.
"""
function densify(S::SparseMatrixCSC, threshold::Real = 0.3)
    m, n = size(S)
    if m == n && isdiag(S)  # convert diagonal sparse to Diagonal
        Diagonal(diag(S))
    elseif nnz(S)/(*(size(S)...)) ≤ threshold # very sparse matrices left as is
        S
    else
        full(S)
    end
end
densify(A::AbstractMatrix, threshold::Real = 0.3) = A

function LinearMixedModel(f, trms, wts)
    n = size(trms[1], 1)
    T = eltype(trms[end])
    sqrtwts = sqrt.(wts)
    if !isempty(wts)
        reweight!.(trms, Vector[sqrtwts])
    end
    nt = length(trms)
    A = Array{AbstractMatrix}(nt, nt)
    L = Array{AbstractMatrix}(nt, nt)
    for j in 1:nt, i in j:nt
        A[i, j] = densify(trms[i]'trms[j])
        L[i, j] = deepcopy(A[i, j])
    end
    for i in 1:nt
        Lii = L[i, i]
        if isa(Lii, Diagonal)
            Liid = Lii.diag
            if !isempty(Liid) && isa(Liid[1], Array)
                if all(d -> size(d) == (1,1), Liid)
                    L[i, i] = Diagonal(map(d -> d[1,1], Liid))
                else
                    L[i, i] = Diagonal(map(LowerTriangular, Liid))
                end
            end
        end
    end
    for i in 2:nt
        Lii = L[i, i]
        if isa(Lii, Diagonal)
            for j in 1:(i - 1)     # check for fill-in
                if !isdiag(A[i, j] * A[i, j]')
                    for k in i:nt
                        L[k, i] = full(L[k, i])
                    end
                end
            end
        end
    end
    optsum = OptSummary(getθ(trms), lowerbd(trms), :LN_BOBYQA;
        ftol_rel = convert(T, 1.0e-12), ftol_abs = convert(T, 1.0e-8))
    fill!(optsum.xtol_abs, 1.0e-10)
    LinearMixedModel(f, trms, sqrtwts, Hermitian(HeteroBlkdMatrix(A), :L),
        LowerTriangular(HeteroBlkdMatrix(L)), optsum)
end

response(typ::DataType, y) = convert(typ, y)
function response(typ::DataType, y::Vector{String})
    if typ <: Vector{<:Number}
        levs = unique(y)
        if length(levs) ≠ 2
            throw(ArgumentError("PooledDataVector y must be binary"))
        end
        return convert(typ, y .== levs[2])
    end
    convert(typ, y)
end

"""
    lmm(f::DataFrames.Formula, fr::DataFrames.DataFrame; weights = [], contrasts = Dict())

Create a [`LinearMixedModel`](@ref) from `f`, which contains both fixed-effects terms
and random effects, and `fr`.

The return value is ready to be `fit!` but has not yet been fit.
"""
function lmm(f::Formula, fr::AbstractDataFrame; weights::Vector = [],
    contrasts= Dict())
    mf = ModelFrame(f, fr, contrasts=contrasts)
    X = ModelMatrix(mf).m
    n = size(X, 1)
    T = eltype(X)
    y = response(Vector{T}, model_response(mf))
    tdict = Dict{Symbol, Vector{Any}}()
    for t in filter(x -> Meta.isexpr(x, :call) && x.args[1] == :|, mf.terms.terms)
        fnm = t.args[3]
        isa(fnm, Symbol) || throw(ArgumentError("rhs of $t must be a symbol"))
        tdict[fnm] = haskey(tdict, fnm) ? push!(tdict[fnm], t.args[2]) : [t.args[2]]
    end
    isempty(tdict) && throw(ArgumentError("No random-effects terms found in $f"))
    trms = AbstractTerm{T}[]
    for (grp, lhs) in tdict
        gr = asfactor(getindex(mf.df, grp))
        m = T[]
        coefnms = String[]
        trsize = Int[]
        for l in lhs
            if l == 1
                append!(m, ones(T, n))
                push!(coefnms, "(Intercept)")
                push!(trsize, 1)
            else
                fr = ModelFrame(Formula(nothing, l), mf.df)
                append!(m, ModelMatrix(fr).m)
                cnms = coefnames(fr)
                append!(coefnms, cnms)
                push!(trsize, length(cnms))
            end
        end
        push!(trms,
            FactorReTerm(gr, transpose(reshape(m, (n, sum(trsize)))), grp, coefnms, trsize))
    end
    sort!(trms, by = nrandomeff, rev = true)
    push!(trms, MatrixTerm(X, coefnames(mf)))
    push!(trms, MatrixTerm(y))
    LinearMixedModel(f, trms, oftype(y, weights))
end

"""
    updateL!(m::LinearMixedModel)

Update the blocked lower Cholesky factor, `m.L`, from `m.A` and `m.trms` (used for Λ only)

This is the crucial step in evaluating the objective, given a new parameter value.
"""
function updateL!{T}(m::LinearMixedModel{T})
    trms = m.trms
    A = m.A.data.blocks
    L = m.L.data.blocks
    nblk = size(A, 2)
    for j in 1:nblk
        Ljj = L[j, j]
        LjjH = isa(Ljj, Diagonal) ? Ljj : Hermitian(Ljj, :L)
        LjjLT = isa(Ljj, Diagonal) ? Ljj : LowerTriangular(Ljj)
        scaleInflate!(Ljj, A[j, j], trms[j])
        for jj in 1:(j - 1)
            rankUpdate!(-one(T), L[j, jj], LjjH)
        end
        cholUnblocked!(Ljj, Val{:L})
        for i in (j + 1):nblk
            Lij = copy!(L[i, j], A[i, j])
            A_mul_Λ!(Lij, trms[j])
            Λc_mul_B!(trms[i], Lij)
            for jj in 1:(j - 1)
                αβA_mul_Bc!(-one(T), L[i, jj], L[j, jj], one(T), Lij)
            end
            A_rdiv_Bc!(Lij, LjjLT)
        end
    end
    m
end

StatsBase.coef(m::LinearMixedModel) = fixef(m)

"""
    fit!(m::LinearMixedModel[, verbose::Bool=false])

Optimize the objective of a `LinearMixedModel`.  When `verbose` is `true` the values of the
objective and the parameters are printed on STDOUT at each function evaluation.
"""
function StatsBase.fit!{T}(m::LinearMixedModel{T}, verbose::Bool=false)
    optsum = m.optsum
    opt = Opt(optsum)
    feval = 0
    function obj(x, g)
        length(g) == 0 || error("gradient not defined")
        feval += 1
        val = objective(updateL!(setθ!(m, x)))
        feval == 1 && (optsum.finitial = val)
        verbose && println("f_", feval, ": ", round(val, 5), " ", x)
        val
    end
    NLopt.min_objective!(opt, obj)
    fmin, xmin, ret = NLopt.optimize!(opt, copy!(optsum.final, optsum.initial))
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
            copy!(xmin, xmin_)
        end
    end
    ## ensure that the parameter values saved in m are xmin
    updateL!(setθ!(m, xmin))

    optsum.feval = feval
    optsum.final = xmin
    optsum.fmin = fmin
    optsum.returnvalue = ret
    ret == :ROUNDOFF_LIMITED && warn("NLopt was roundoff limited")
    if ret ∈ [:FAILURE, :INVALID_ARGS, :OUT_OF_MEMORY, :FORCED_STOP]
        warn("NLopt optimization failure: $ret")
    end
    m
end

function fitted!{T}(v::AbstractArray{T}, m::LinearMixedModel{T})
    ## FIXME: Create and use `effects(m) -> β, b` w/o calculating β twice
    trms = m.trms
    A_mul_B!(vec(v), trms[end - 1], fixef(m))
    b = ranef(m)
    for j in eachindex(b)
        unscaledre!(vec(v), trms[j], b[j])
    end
    v
end

StatsBase.fitted{T}(m::LinearMixedModel{T}) = fitted!(Vector{T}(nobs(m)), m)

StatsBase.residuals(m::LinearMixedModel) = model_response(m) .- fitted(m)

"""
    lowerbd(m::LinearMixedModel)

Return the vector of lower bounds on the covariance parameter vector `θ`
"""
lowerbd{T}(m::LinearMixedModel{T}) = mapreduce(lowerbd, append!, T[], m.trms)

"""
    objective(m::LinearMixedModel)

Return negative twice the log-likelihood of model `m`
"""
objective(m::LinearMixedModel) = logdet(m) + nobs(m) * (1 + log2π + log(varest(m)))

"""
    fixef!{T}(v::Vector{T}, m::LinearMixedModel{T})

Overwrite `v` with the fixed-effects coefficients of model `m`
"""
function fixef!{T}(v::AbstractVector{T}, m::LinearMixedModel{T})
    !isfit(m) && throw(ArgumentError("Model m has not been fit"))
    Ac_ldiv_B!(feL(m), copy!(v, m.L[end, end - 1]))
end

"""
    fixef(m::MixedModel)

Returns the fixed-effects parameter vector estimate.
"""
fixef{T}(m::LinearMixedModel{T}) = fixef!(Array{T}(size(m)[2]), m)

StatsBase.dof(m::LinearMixedModel) = size(m.trms[end - 1].wtx, 2) + sum(nθ, m.trms) + 1

StatsBase.loglikelihood(m::LinearMixedModel) = -deviance(m)/2

StatsBase.nobs(m::LinearMixedModel) = length(m.trms[end].wtx)

function Base.size(m::LinearMixedModel)
    trms = m.trms
    n, p = size(trms[end - 1])
    k = length(trms) - 2
    q = sum(size(trms[j], 2) for j in 1:k)
    n, p, q, k
end

"""
    sdest(m::LinearMixedModel)

Return the estimate of σ, the standard deviation of the per-observation noise.
"""
sdest(m::LinearMixedModel) = sqrtpwrss(m) / √nobs(m)

"""
    setθ!{T}(m::LinearMixedModel{T}, v::Vector{T})

Install `v` as the θ parameters in `m`.
"""
function setθ!{T}(m::LinearMixedModel{T}, v::Vector{T})
    setθ!(m.trms, v)
    m
end

"""
    sqrtpwrss(m::LinearMixedModel)

Return the square root of the penalized, weighted residual sum-of-squares (pwrss).

This value is the contents of the `1 × 1` bottom right block of `m.L`
"""
sqrtpwrss(m::LinearMixedModel) = @views m.L[end, end][1]

"""
    varest(m::LinearMixedModel)

Returns the estimate of σ², the variance of the conditional distribution of Y given B.
"""
varest(m::LinearMixedModel) = pwrss(m) / nobs(m)

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
    if (nm = length(mods)) <= 1
        throw(ArgumentError("at least two models are required for a likelihood ratio test"))
    end
    m1 = mods[1]
    n = nobs(m1)
    for i in 2:nm
        if nobs(mods[i]) != n
            throw(ArgumentError("number of observations must be constant across models"))
        end
    end
    mods = mods[sortperm([dof(m)::Int for m in mods])]
    degf = Int[dof(m) for m in mods]
    dev = [deviance(m)::Float64 for m in mods]
    csqr = unshift!([(dev[i-1]-dev[i])::Float64 for i in 2:nm],NaN)
    pval = unshift!([ccdf(Chisq(degf[i]-degf[i-1]),csqr[i])::Float64 for i in 2:nm],NaN)
    DataFrame(Df = degf, Deviance = dev, Chisq=csqr,pval=pval)
end

"""
    reweight!{T}(m::LinearMixedModel{T}, wts::Vector{T})

Update `m.sqrtwts` from `wts` and reweight each term.  Recompute `m.A` and `m.L`.
"""
function reweight!{T}(m::LinearMixedModel{T}, weights::Vector{T})
    trms = m.trms
    sqrtwts = m.sqrtwts
    @argcheck(length(weights) == length(sqrtwts), DimensionMismatch)
    map!(sqrt, sqrtwts, weights)
    reweight!.(trms, Vector{T}[sqrtwts])
    ntrm = length(trms)
    A = m.A
    for j in 1:ntrm, i in j:ntrm
        Ac_mul_B!(A[i, j], trms[i], trms[j])
    end
    updateL!(m)
end

function Base.show(io::IO, m::LinearMixedModel)
    if !isfit(m)
        warn("Model has not been fit")
        return nothing
    end
    n, p, q, k = size(m)
    println(io, "Linear mixed model fit by maximum likelihood")
    println(io, " ", m.formula)
    oo = objective(m)
    nums = showoff([-oo/ 2, oo, aic(m), bic(m)])
    fieldwd = max(maximum(strwidth.(nums)) + 1, 11)
    for label in [" logLik", "-2 logLik", "AIC", "BIC"]
        print(io, Base.cpad(label, fieldwd))
    end
    println(io)
    for num in nums
        print(io, lpad(num, fieldwd))
    end
    println(io); println(io)

    show(io,VarCorr(m))

    gl = grplevels(m)
    @printf(io," Number of obs: %d; levels of grouping factors: %d", n, gl[1])
    for l in gl[2:end] @printf(io, ", %d", l) end
    println(io)
    @printf(io,"\n  Fixed-effects parameters:\n")
    show(io,coeftable(m))
end
