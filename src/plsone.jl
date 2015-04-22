## Solver for models with a single grouping factor for the random effects

## There are l₁ levels in the grouping factor.  The dimension of the
## random effects for each level of the grouping factor is l₁,
## producing a total of q = p₁l₁ random effects.  The dimension of the
## fixed-effects parameter is p.  When solving for the conditional
## modes of U only the transposed model matrix Xt is passed as a 0×n
## matrix.

## Within the type:
## Z'Z is block diagonal with l₁ blocks of size p₁×p₁, stored as A₁₁, a p₁×(p₁l₁) matrix
## X'Z is stored as A₂₁, a p×(p₁l₁) matrix
## X'X is stored as A₂₂, a p×p matrix - it is symmetric but not explicitly stored as Symmetric
## L₁₁ is block diagonal with l₁ lower triangular blocks of size p₁×p₁
## L₂₁ is stored as a matrix similar to A₂₁
## L₂₂ is stored as a p×p lower Cholesky factor

type PLSOne <: PLSSolver
    A₁₁::Matrix{Float64}
    A₂₁::Matrix{Float64}
    A₂₂::Matrix{Float64}
    L₁₁::Matrix{Float64}
    L₂₁::Matrix{Float64}
    L₂₂::Cholesky{Float64}
    gtmp::Vector{Matrix{Float64}}
end

function PLSOne(ff::PooledDataVector, Xst::Matrix, Xt::Matrix)
    refs = ff.refs
    (n = length(refs)) == size(Xst,2) == size(Xt,2) || throw(DimensionMismatch("PLSOne"))
    p = size(Xt,1)               # number of fixed-effects parameters
    p₁ = size(Xst,1)             # number of random effects per level
    l₁ = length(ff.pool)         # number of levels of grouping factor
    q₁ = p₁*l₁                   # total number of random effects
    A₁₁ = zeros(p₁,q₁)
    A₂₁ = zeros(p,q₁)
    for j in 1:n
        i1 = refs[j]-1
        c₁ = ContiguousView(Xst,p₁*(j-1),(p₁,))
        BLAS.ger!(1.0,c₁,c₁,ContiguousView(A₁₁,i1*p₁*p₁,(p₁,p₁)))
        BLAS.ger!(1.0,ContiguousView(Xt,p*(j-1),(p,)),c₁,ContiguousView(A₂₁,i1*p*p₁,(p,p₁)))
    end
    PLSOne(A₁₁,A₂₁,tril!(Xt*Xt'),similar(A₁₁),similar(A₂₁),cholfact(eye(p),:L),
           Matrix{Float64}[zeros(p₁,p₁)])
end

## argument uβ contains vcat(λ'Z'y,X'y) on entry
function Base.A_ldiv_B!(s::PLSOne,uβ)
    p,p₁,l₁ = size(s)
    q₁ = p₁*l₁
    length(uβ) == (p + q₁) || throw(DimensionMismatch(""))
    u = ContiguousView(uβ,(q₁,))
    β = ContiguousView(uβ,q₁,(p,))
    if p₁ == 1                          # short cut for scalar r.e.
        scaleinv!(u,vec(s.L₁₁))
        A_ldiv_B!(s.L₂₂,BLAS.gemv!('N',-1.,s.L₂₁,u,1.,β)) # solve for β
        BLAS.gemv!('T',-1.,s.L₂₁,β,1.0,u) # cᵤ -= L₂₁'β
        scaleinv!(u,vec(s.L₁₁))
    else
        for j in 0:(l₁-1) # solve L cᵤ = λ'Z'y blockwise using offsets
            BLAS.trsv!('L','N','N',ContiguousView(s.L₁₁,j*p₁*p₁,(p₁,p₁)),
                       ContiguousView(uβ,j*p₁,(p₁,)))
        end
                                        # solve (L_X L_X')̱β = X'y - L_XZ cᵤ
        A_ldiv_B!(s.L₂₂,BLAS.gemv!('N',-1.0,s.L₂₁,u,1.0,β))
                                        # cᵤ := cᵤ - L_XZ'β
        BLAS.gemv!('T',-1.0,reshape(s.L₂₁,(p,q₁)),β,1.0,u)
        for j in 0:(l₁-1) # # solve L'u = cᵤ blockwise using offsets
            BLAS.trsv!('L','T','N',ContiguousView(s.L₁₁,j*p₁*p₁,(p₁,p₁)),
                       ContiguousView(uβ,j*p₁,(p₁,)))
        end
    end
end

function Base.cholfact(s::PLSOne,RX::Bool=true)
    RX && return s.L₂₂
    p,p₁,l₁ = size(s)
    blkdiag([sparse(tril(ContiguousView(s.L₁₁,(j-1)*p₁*p₁,(p₁,p₁)))) for j in 1:l₁]...)
end

## grad calculation - evaluates the sum of the diagonal blocks of (LL')⁻¹*Λ'*Z'Z
## The function is mutating on its first argument which, on entry, contains the gradient
## blocks from the penalized residual sum of squares term.

function grad!(res::Vector{Matrix{Float64}},s::PLSOne,λ::Vector)
    length(λ) == 1 || throw(DimensionMismatch(""))
    λ = λ[1]
    _,p₁,ℓ₁ = size(s)
    r₁ = res[1]
    t₁ = s.gtmp[1]
    for i in inds(p₁,ℓ₁)
        LAPACK.potrs!('L',view(s.L₁₁,:,i),Ac_mul_B!(λ,copy!(t₁,view(s.A₁₁,:,i))))
        for j in 1:abs2(p₁)
            r₁[j] += t₁[j]
        end
    end
    r₁
end

## Logarithm of the determinant of the matrix represented by RX or L
function Base.logdet(s::PLSOne,RX=true)
    RX && return logdet(s.L₂₂)
    p,p₁,l₁ = size(s)
    sm = 0.
    for j in 0:(l₁-1), i in 1:p₁
        sm += log(s.L₁₁[i,j*p₁+i])
    end
    2.sm
end

function Base.size(s::PLSOne)
    p₁,q₁ = size(s.A₁₁)
    q₁ % p₁ == 0 || throw(DimensionMismatch(""))
    size(s.A₂₁,1), p₁, div(q₁,p₁)
end

##  update!(s,λ)->s : update L₁₁, L₂₁ and L₂₂
function update!(s::PLSOne,λ::Vector)
    length(λ) == 1 || error("update! on a PLSOne requires length(λ) == 1")
    λ = λ[1]
    isa(λ,AbstractPDMatFactor) || error("λ must be a vector of PDMatFactors")
    dim(λ) == size(s.A₁₁,1) || throw(DimensionMixmatch(""))
    p,p₁,l₁ = size(s)
    L₂₁ = copy!(s.L₂₁,s.A₂₁)
    if p == 1                           # shortcut for 1×1 λ
        isa(λ,PDScalF) || error("1×1 λ section should be a PDScalF type")
        lam = λ.s
        lamsq = lam*lam
        for j in 1:l₁
            s.L₁₁[1,j] = sqrt(s.A₁₁[1,j]*lamsq + 1.)
            scale!(view(L₂₁,:,j),lam/s.L₁₁[1,j])
        end
    else
        Ac_mul_B!(λ,copy!(s.L₁₁,s.A₁₁))   # multiply on left by λ'
        cols = 1:p₁
        for k in 1:l₁
            wL = A_mul_B!(view(s.L₁₁,:,cols),λ)
            for j in 1:p₁               # inflate the diagonal
                wL[j,j] += 1.0
            end
            _,info = LAPACK.potrf!('L',wL) # lower Cholesky factor
            info == 0 || error("Cholesky failure at L diagonal block $(k+1)")
            A_rdiv_Bc!(A_mul_B!(view(L₂₁,:,cols),λ),ltri(wL))
            cols += p₁
        end
    end
    BLAS.syrk!('L','N',-1.,L₂₁,1.,copy!(s.L₂₂.UL,s.A₂₂))
    _, info = LAPACK.potrf!('L',s.L₂₂.UL)
    info == 0 ||  error("downdated X'X is not positive definite")
    s
end
