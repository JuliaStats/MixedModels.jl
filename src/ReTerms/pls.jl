"""
Summary of optimization results using the NLopt package
"""
type OptSummary
    initial::Vector{Float64}
    final::Vector{Float64}
    fmin::Float64
    feval::Int
    geval::Int
    optimizer::Symbol
end
function OptSummary(initial::Vector{Float64},optimizer::Symbol)
    OptSummary(initial,initial,Inf,-1,-1,optimizer)
end

"""
Linear mixed-effects model representation

`trms` is a length `nt` vector of model matrices whose last element is `hcat(X,y)`
`Λ` is a length `nt - 1` vector of parameterized lower triangular matrices
`A` is an `nt × nt` symmetric matrix of matrices representing `hcat(Z,X,y)'hcat(Z,X,y)`
`R`, also a `nt × nt` matrix of matrices, is the upper Cholesky factor of `Λ'AΛ`
"""
type LinearMixedModel <: MixedModel
    trms::Vector
    Λ::Vector
    A::Matrix        # symmetric cross-product blocks (upper triangle)
    R::Matrix        # right Cholesky factor in blocks.
    opt::OptSummary
end

function LinearMixedModel(Rem::Vector, Λ::Vector, X::AbstractMatrix, y::Vector)
    all(x->isa(x,ReMat),Rem) ||
        throw(ArgumentError("Elements of Rem should be ReMat's"))
    all(x->isa(x,ParamLowerTriangular),Λ) ||
        throw(ArgumentError("Elements of Λ should be ParamLowerTriangular"))
    n,p = size(X)
    all(t -> size(t,1) == n,Rem) && length(y) == n || throw(DimensionMismatch("n not consistent"))
    nreterms = length(Rem)
    length(Λ) == nreterms && all(i->chksz(Rem[i],Λ[i]),1:nreterms) ||
        throw(DimensionMismatch("Rem and Λ"))
    nt = nreterms + 1
    trms = Array(Any,nt)
    for i in eachindex(Rem) trms[i] = Rem[i] end
    trms[end] = hcat(X,y)
    A = fill!(Array(Any,(nt,nt)),nothing)
    R = fill!(Array(Any,(nt,nt)),nothing)
    for j in 1:nt
        for i in 1:j
            A[i,j] = densify(trms[i]'trms[j])
            R[i,j] = copy(A[i,j])
        end
    end
    for j in 2:nreterms
        if isa(R[j,j],Diagonal) || isa(R[j,j],HBlkDiag)
            for i in 1:(j-1)     # check for fill-in
                if !isdiag(A[i,j]'A[i,j])
                    for k in j:nt
                        R[j,k] = full(R[j,k])
                    end
                end
            end
        end
    end
    LinearMixedModel(trms,Λ,A,R,OptSummary(mapreduce(x->x[:θ],vcat,Λ),:None))
end

LinearMixedModel(re::Vector,Λ::Vector,X::AbstractMatrix,y::DataVector) = LinearMixedModel(re,Λ,X,convert(Array,y))

LinearMixedModel(re::Vector,X::DenseMatrix,y::DataVector) = LinearMixedModel(re,map(LT,re),X,convert(Array,y))

LinearMixedModel(re::Vector,X::DenseMatrix,y::Vector) = LinearMixedModel(re,map(LT,re),X,y)

LinearMixedModel(g::PooledDataVector,y::DataVector) = LinearMixedModel([ReMat(g)],y)

LinearMixedModel(re::Vector,y::DataVector) = LinearMixedModel(re,ones(length(y),1),y)

LinearMixedModel(re::Vector,y::Vector) = LinearMixedModel(re,map(LT,re),ones(length(y),1),y)

function LinearMixedModel(f::Formula, fr::AbstractDataFrame)
    mf = ModelFrame(f,fr)
    X = ModelMatrix(mf)
    y = convert(Vector{Float64},DataFrames.model_response(mf))
                                        # process the random-effects terms
    retrms = filter(x->Meta.isexpr(x,:call) && x.args[1] == :|, mf.terms.terms)
    length(retrms) > 0 || error("Formula $f has no random-effects terms")
    re = [remat(e,mf.df) for e in retrms]
    LinearMixedModel(re,X.m,y)
end


chksz(A::ReMat,λ::ParamLowerTriangular) = size(λ,1) == 1
chksz(A::VectorReMat,λ::ParamLowerTriangular) = size(λ,1) == size(A.z,1)

lowerbd(A::LinearMixedModel) = mapreduce(lowerbd,vcat,A.Λ)

Base.getindex(m::LinearMixedModel,s::Symbol) = mapreduce(x->x[s],vcat,m.Λ)

function Base.setindex!(m::LinearMixedModel,v::Vector,s::Symbol)
    s == :θ || throw(ArgumentError("only ':θ' is meaningful for assignment"))
    lam = m.Λ
    length(v) == sum(nθ,lam) || throw(DimensionMismatch("length(v) should be $(sum(nθ,lam))"))
    A = m.A
    R = m.R
    n = size(A,1)                       # inject upper triangle of A into L
    for j in 1:n, i in 1:j
        inject!(R[i,j],A[i,j])
    end
    offset = 0
    for i in eachindex(lam)
        li = lam[i]
        nti = nθ(li)
        li[:θ] = sub(v,offset + (1:nti))
        offset += nti
        for j in i:size(R,2)
            scale!(li,R[i,j])
        end
        for ii in 1:i
            scale!(R[ii,i],li)
        end
        inflate!(R[i,i])
    end
    cfactor!(R)
end

"""
`fit(m)` -> `m`

Optimize the objective using an NLopt optimizer.
"""
function StatsBase.fit(m::LinearMixedModel, verbose::Bool=false, optimizer::Symbol=:default)
    th = m[:θ]
    k = length(th)
    if optimizer == :default
        optimizer = hasgrad(m) ? :LD_MMA : :LN_BOBYQA
    end
    opt = NLopt.Opt(optimizer, k)
    NLopt.ftol_rel!(opt, 1e-12)   # relative criterion on deviance
    NLopt.ftol_abs!(opt, 1e-8)    # absolute criterion on deviance
    NLopt.xtol_abs!(opt, 1e-10)   # criterion on parameter value changes
    NLopt.lower_bounds!(opt, lowerbd(m))
    feval = 0
    geval = 0
    function obj(x::Vector{Float64}, g::Vector{Float64})
        feval += 1
        m[:θ] = x
        val = objective(m)
        if length(g) == length(x)
            geval += 1
            grad!(g,m)
        end
        val
    end
    if verbose
        function vobj(x::Vector{Float64}, g::Vector{Float64})
            feval += 1
            m[:θ] = x
            val = objective(m)
            print("f_$feval: $(round(val,5)), [")
            showcompact(x[1])
            for i in 2:length(x) print(","); showcompact(x[i]) end
            println("]")
            if length(g) == length(x)
                geval += 1
                grad!(g,m)
            end
            val
        end
        NLopt.min_objective!(opt, vobj)
    else
        NLopt.min_objective!(opt, obj)
    end
    fmin, xmin, ret = NLopt.optimize(opt, th)
    ## very small parameter values often should be set to zero
    xmin1 = copy(xmin)
    modified = false
    for i in eachindex(xmin1)
        if 0. < abs(xmin1[i]) < 1.e-5
            modified = true
            xmin1[i] = 0.
        end
    end
    if modified
        m[:θ] = xmin1
        if (ff = objective(m)) < fmin
            fmin = ff
            copy!(xmin,xmin1)
        else
            m[:θ] = xmin
        end
    end
    m.opt = OptSummary(th,xmin,fmin,feval,geval,optimizer)
    if verbose println(ret) end
    m
end

grad!(v,lmm::LinearMixedModel) = v

hasgrad(lmm::LinearMixedModel) = false

"""
Regenerate the last column of `m.A` from `m.trms`

This should be called after updating parts of `m.trms[end]`, typically the response.
"""
function regenerateAend!(m::LinearMixedModel)
    n = Base.LinAlg.chksquare(m.A)
    trmn = m.trms[n]
    for i in 1:n
        Ac_mul_B!(m.A[i,n],m.trms[i],trmn)
    end
    m
end

"""
Reset the value of `m.θ` to the initial values
"""
function resetθ!(m::LinearMixedModel)
    m[:θ] = m.opt.initial
    m.opt.feval = m.opt.geval = -1
    m
end

"Add an identity matrix to the argument, in place"
inflate!(D::Diagonal{Float64}) = (d = D.diag; for i in eachindex(d) d[i] += 1 end; D)

function inflate!{T<:AbstractFloat}(A::StridedMatrix{T})
    n = Base.LinAlg.chksquare(A)
    for i in 1:n
        @inbounds A[i,i] += 1
    end
    A
end

"""
LD(A) -> log(det(A)) for A diagonal, HBlkDiag, or UpperTriangular
"""
function LD{T}(d::Diagonal{T})
    r = log(one(T))
    dd = d.diag
    for i in eachindex(dd)
        r += log(dd[i])
    end
    r
end

function LD{T}(d::HBlkDiag{T})
    r = log(one(T))
    aa = d.arr
    p,q,k = size(aa)
    for j in 1:k, i in 1:p
        r += log(aa[i,i,j])
    end
    r
end

function LD{T}(d::DenseMatrix{T})
    r = log(one(T))
    n = Base.LinAlg.chksquare(d)
    for j in 1:n
        r += log(d[j,j])
    end
    r
end

function LD(m::LinearMixedModel)
    R = m.R
    s = 0.
    for i in eachindex(m.Λ)
        s += LD(R[i,i])
    end
    2.*s
end

"Negative twice the log-likelihood"
objective(m::LinearMixedModel) = LD(m) + nobs(m)*(1.+log(2π*varest(m)))

function Base.LinAlg.Ac_ldiv_B!{T}(D::UpperTriangular{T,Diagonal{T}},B::DenseMatrix{T})
    m,n = size(B)
    dd = D.data.diag
    length(dd) == m || throw(DimensionMismatch(""))
    for j in 1:n, i in 1:m
        B[i,j] /= dd[i]
    end
    B
end

function Base.LinAlg.Ac_ldiv_B!{T}(A::UpperTriangular{T,HBlkDiag{T}},B::DenseMatrix{T})
    m,n = size(B)
    aa = A.data.arr
    r,s,k = size(aa)
    m == Base.LinAlg.chksquare(A) || throw(DimensionMismatch())
    scr = Array(T,(r,n))
    for i in 1:k
        bb = sub(B,(i-1)*r+(1:r),:)
        copy!(bb,Base.LinAlg.Ac_ldiv_B!(UpperTriangular(sub(aa,:,:,i)),copy!(scr,bb)))
    end
    B
end

function Base.LinAlg.Ac_ldiv_B!{T}(D::UpperTriangular{T,Diagonal{T}},B::SparseMatrixCSC{T})
    m,n = size(B)
    dd = D.data.diag
    length(dd) == m || throw(DimensionMismatch(""))
    nzv = nonzeros(B)
    rv = rowvals(B)
    for j in 1:n, k in nzrange(B,j)
        nzv[k] /= dd[rv[k]]
    end
    B
end

Base.LinAlg.A_ldiv_B!{T<:AbstractFloat}(D::Diagonal{T},B::DenseMatrix{T}) =
    Base.LinAlg.Ac_ldiv_B!(D,B)

function Base.logdet(t::UpperTriangular)
    n = Base.LinAlg.chksquare(t)
    mapreduce(log,(+),diag(t))
end

function Base.logdet(t::LowerTriangular)
    n = Base.LinAlg.chksquare(t)
    mapreduce(log,(+),diag(t))
end

function Base.logdet{T<:AbstractFloat}(R::HBlkDiag{T})
    ra = R.arr
    ret = zero(T)
    r,s,k = size(ra)
    for i in 1:k
        ret += logdet(UpperTriangular(sub(ra,:,:,i)))
    end
    ret
end

function Base.LinAlg.A_rdiv_Bc!{T<:AbstractFloat}(A::SparseMatrixCSC{T},B::Diagonal{T})
    m,n = size(A)
    dd = B.diag
    n == length(dd) || throw(DimensionMismatch(""))
    nz = nonzeros(A)
    for j in eachindex(dd)
        @inbounds scale!(sub(nz,nzrange(A,j)),inv(dd[j]))
    end
    A
end

function Base.LinAlg.A_rdiv_Bc!{T<:AbstractFloat}(A::Matrix{T},B::Diagonal{T})
    m,n = size(A)
    dd = B.diag
    n == length(dd) || throw(DimensionMismatch(""))
    for j in eachindex(dd)
        @inbounds scale!(sub(A,:,j),inv(dd[j]))
    end
    A
end

function Base.LinAlg.A_rdiv_B!(A::StridedVecOrMat,D::Diagonal)
    m, n = size(A, 1), size(A, 2)
    if n != length(D.diag)
        throw(DimensionMismatch("diagonal matrix is $(length(D.diag)) by $(length(D.diag)) but left hand side has $n columns"))
    end
    (m == 0 || n == 0) && return A
    dd = D.diag
    for j = 1:n
        dj = dd[j]
        if dj == 0
            throw(SingularException(j))
        end
        for i = 1:m
            A[i,j] /= dj
        end
    end
    A
end

AIC(m::LinearMixedModel) = objective(m) + 2npar(m)

BIC(m::LinearMixedModel) = objective(m) + npar(m)*log(nobs(m))

Base.LinAlg.A_rdiv_Bc!(A::StridedVecOrMat,D::Diagonal) = A_rdiv_B!(A,D)

## Rename this
Base.cholfact(m::LinearMixedModel) = UpperTriangular(m.R[end,end][1:end-1,1:end-1])

fixef(m::LinearMixedModel) = cholfact(m)\m.R[end,end][1:end-1,end]

StatsBase.coef(m::LinearMixedModel) = fixef(m)

StatsBase.nobs(m::LinearMixedModel) = size(m.trms[end],1)

function StatsBase.vcov(m::LinearMixedModel)
    Rinv = Base.LinAlg.inv!(cholfact(m))
    varest(m)*Rinv*Rinv'
end

"""
Number of parameters in the model.

Note that `size(m.trms[end],2)` is `length(coef(m)) + 1`, thereby accounting
for the scale parameter, σ, that is profiled out.
"""
npar(m::LinearMixedModel) = size(m.trms[end],2) + length(m[:θ])

"""
returns the square root of the penalized residual sum-of-squares

This is the bottom right element of the bottom right block of m.R
"""
sqrtpwrss(m::LinearMixedModel) = m.R[end,end][end,end]

"""
returns s², the estimate of σ², the variance of the conditional distribution of Y given B
"""
varest(m::LinearMixedModel) = pwrss(m)/nobs(m)

"""
returns the penalized residual sum-of-squares
"""
pwrss(m::LinearMixedModel) = abs2(sqrtpwrss(m))

"""
Simulate `N` response vectors from `m`, refitting the model.  The function saveresults
is called after each refit.

To save space the last column of `m.trms[end]`, which is the response vector, is overwritten
by each simulation.  The original response is restored before returning.
"""
function bootstrap(m::LinearMixedModel,N::Integer,saveresults::Function,σ::Real=-1,mods::Vector{LinearMixedModel}=LinearMixedModel[])
    if σ < 0.
        σ = √varest(m)
    end
    trms = m.trms
    nt = length(trms)
    Xy = trms[nt]
    n,pp1 = size(Xy)
    X = sub(Xy,:,1:pp1-1)
    y = sub(Xy,:,pp1)
    y0 = copy(y)
    fev = X*coef(m)
    vfac = [σ*convert(LowerTriangular,λ) for λ in m.Λ]  # lower Cholesky factors of relative covariances
    remats = Matrix{Float64}[]
    for i in eachindex(vfac)
        vsz = size(trms[i],2)
        nr = size(vfac[i],1)
        push!(remats,reshape(zeros(vsz),(nr,div(vsz,nr))))
    end
    for kk in 1:N
        for i in 1:n
            y[i] = fev[i] + σ*randn()
        end
        for j in eachindex(remats)
            mat = vec(A_mul_B!(vfac[j],randn!(remats[j])))
            A_mul_B!(1.,trms[j],vec(mat),1.,y)
        end
        regenerateAend!(m)
        resetθ!(m)
        fit(m)
        saveresults(kk,m)
    end
    copy!(y,y0)
    regenerateAend!(m)
    resetθ!(m)
    fit(m)
end
