## Solver for models with two crossed or nearly-crossed grouping factors for the random effects

## There are ℓ₁ and ℓ₂ levels in the grouping factors.  The dimension
## of the random effects vector for each level of the grouping factors is p₁
## and p₂, respectively. The total number of random effects is q₁+q₂
## where qᵢ=pᵢℓᵢ, i=1,2.  The dimension of the fixed-effects parameter
## is p.  When solving for the conditional modes of U only the
## transposed model matrix, Xt, passed to the constructor has 0 rows.

## Within the type:
## Z₁'Z₁ is block diagonal with ℓ₁ blocks of size p₁×p₁, stored as A₁₁, a p₁×q₁ matrix
## Z₂'Z₁ is stored as A₂₁, a q₂×q₁ matrix.  The fraction of zeros in A₂₁ should be small.
## X'Z₁ is stored as A₃₁, a p×q₁ matrix.
## Z₂'Z₂ is block diagonal with ℓ₂ symmetric blocks of size p₂×p₂, stored as A₂₂, a p₁×q₁ matrix
## X'Z₂ is stored as A₃₂, a p×q₂ matrix.
## X'X is stored as A₃₃, a p×p matrix - symmetric but not explicitly a Symmetric type
## L₁₁ is block diagonal with ℓ₁ lower triangular blocks of size p₁×p₁, stored as a p₂×q₂ matrix
## L₂₁ is stored as a q₂×q₁ matrix
## L₃₁ is stored as a p×q₁ matrix
## L₂₂ is stored as a lower Triangular q₂×q₂ matrix, due to fill-in
## L₃₂ is stored as a p×q₂ matrix
## L₃₃ is stored as a p×p lower Cholesky factor

type PLSTwo <: PLSSolver # Solver for models with two crossed or nearly crossed terms
    A₁₁::Matrix{Float64} # diagonal blocks of Z₁'Z₁
    A₂₁::Matrix{Float64} # Z₂'*Z₁
    A₃₁::Matrix{Float64} # X'Z₁
    A₂₂::Matrix{Float64} # diagonal blocks of Z₂'Z₂
    A₃₂::Matrix{Float64} # X'Z₂
    A₃₃::Matrix{Float64} # X'X
    L₁₁::Matrix{Float64} 
    L₂₁::Matrix{Float64}
    L₃₁::Matrix{Float64}
    L₂₂::Triangular{Float64,Matrix{Float64},:L,false} 
    L₃₂::Matrix{Float64}
    L₃₃::Cholesky{Float64}
end

function PLSTwo(facs::Vector,Xst::Vector,Xt::Matrix)
    length(facs) == length(Xst) == 2 || throw(DimensionMismatch("PLSTwo"))
                                        # check for consistency in number of observations
    (n = size(Xt,2)) == size(Xst[1],2) == size(Xst[2],2) ==
        length(facs[1]) == length(facs[2]) || throw(DimensionMismatch(""))
    l₁ = length(facs[1].pool)
    l₂ = length(facs[2].pool)
    p₁ = size(Xst[1],1)
    p₂ = size(Xst[2],1)
    q₁ = p₁ * l₁
    q₂ = p₂ * l₂
    p = size(Xt,1)
                                        # Do this in lmm instead of checking here
    q₁ ≥ q₂ || error("reverse the order of the random effects terms")
    A₁₁ = zeros(p₁,q₁)
    A₂₁ = zeros(q₂,q₁)
    A₃₁ = zeros(p,q₁)
    A₂₂ = zeros(p₂,q₂)
    A₃₂ = zeros(p,q₂)
    r₁ = facs[1].refs
    r₂ = facs[2].refs
    inds₁ = 1:p₁
    inds₂ = 1:p₂
    for j in 1:n
        i₁ = (r₁[j] - 1)*p₁ + inds₁
        i₂ = (r₂[j] - 1)*p₂ + inds₂
        c₁ = view(Xst[1],:,j)
        c₂ = view(Xst[2],:,j)
        c₃ = view(Xt,:,j)
        ## some of these could be syr! but we want the result to be symmetric
        BLAS.ger!(1.,c₁,c₁,view(A₁₁,:,i₁))
        BLAS.ger!(1.,c₂,c₁,view(A₂₁,i₂,i₁))
        BLAS.ger!(1.,c₃,c₁,view(A₃₁,:,i₁))
        BLAS.ger!(1.,c₂,c₂,view(A₂₂,:,i₂))
        BLAS.ger!(1.,c₃,c₂,view(A₃₂,:,i₂))
    end
    PLSTwo(A₁₁,A₂₁,A₃₁,A₂₂,A₃₂,Xt*Xt',similar(A₁₁),similar(A₂₁),similar(A₃₁),
           Triangular(zeros(q₂,q₂),:L,false),similar(A₃₂),cholfact!(eye(p),:L))
end

function Base.cholfact(s::PLSTwo,RX::Bool=true)
    RX && return s.L₃₃
    p,p₁,p₂,ℓ₁,ℓ₂,q₁,q₂ = size(s)
    L₁₁ = blkdiag({sparse(tril(view(s.L₁₁,:,i₁))) for i₁ in inds(p₁,ℓ₁)}...)
    vcat(hcat(L₁₁,spzeros(q₁,q₂)),sparse(hcat(s.L₂₁,s.L₂₂)))
end

function Base.logdet(s::PLSTwo,RX=true)
    RX && return logdet(s.L₃₃)
    p,p₁,p₂,ℓ₁,ℓ₂,q₁,q₂ = size(s)
    sm = 0.
    for i₁ in inds(p₁,ℓ₁)
        L₁₁ = view(s.L₁₁,:,i₁)
        for j in 1:p₁
            sm += log(L₁₁[j,j])
        end
    end
    for j in 1:q₂
        sm += log(s.L₂₂[j,j])
    end
    2.sm
end

function Base.A_ldiv_B!(s::PLSTwo,uβ)
    p,p₁,p₂,ℓ₁,ℓ₂,q₁,q₂ = size(s)
    length(uβ) == p+q₁+q₂ || throw(DimensionMismatch(""))
    u₁ = view(uβ,1:q₁)
    u₂ = view(uβ,q₁+(1:q₂))
    β = view(uβ,q₁+q₂+(1:p))
    if p₁ == 1                          # scalar r.e. for factor 1
        scaleinv!(u₁,vec(s.L₁₁))
    else
        for i₁ in inds(p₁,ℓ₁)           # solve L₁ cᵤ₁ = λ₁'Z₁'y blockwise
            A_ldiv_B!(Triangular(view(s.L₁₁,:,i₁),:L,false),view(uβ,i₁))
        end
    end
    A_ldiv_B!(s.L₂₂,BLAS.gemv!('N',-1.,s.L₂₁,u₁,1.,u₂))
    A_ldiv_B!(s.L₃₃,BLAS.gemv!('N',-1.,s.L₃₂,u₂,1.,BLAS.gemv!('N',-1.,s.L₃₁,u₁,1.,β)))
    Ac_ldiv_B!(s.L₂₂,BLAS.gemv!('T',-1.,s.L₃₂,β,1.,u₂))
    BLAS.gemv!('T',-1.,s.L₃₁,β,1.,BLAS.gemv!('T',-1.,s.L₂₁,u₂,1.,u₁))
    if p₁ == 1
        scaleinv!(u₁,vec(s.L₁₁))
    else
        for i₁ in inds(p₁,ℓ₁)           # solve L₁₁'u₁ = cᵤ₁ blockwise
            Ac_ldiv_B!(Triangular(view(s.L₁₁,:,i₁),:L,false),view(uβ,i₁))
        end
    end
    uβ
end

function Base.size(s::PLSTwo)
    p₁,q₁ = size(s.A₁₁)
    q₁ % p₁ == 0 || throw(DimensionMismatch(""))
    p₂,q₂ = size(s.A₂₂)
    q₂ % p₂ == 0 || throw(DimensionMismatch(""))
    size(s.A₃₃,1),p₁,p₂,div(q₁,p₁),div(q₂,p₂),q₁,q₂
end

function update!(s::PLSTwo,λ::Vector)
    length(λ) == 2 || throw(DimensionMismatch(""))
    for ll in λ
        isa(ll,AbstractPDMatFactor) || error("λ must be a vector of PDMatFactors")
    end
    λ₁ = λ[1]
    λ₂ = λ[2]    
    p,p₁,p₂,ℓ₁,ℓ₂,q₁,q₂ = size(s)
    (dim(λ₁) == p₁ && dim(λ₂) == p₂) || throw(DimensionMismatch(""))
    L₂₁ = copy!(s.L₂₁,s.A₂₁)
    L₃₁ = copy!(s.L₃₁,s.A₃₁)
    if p₁ == 1                          # shortcut for 1×1 λ
        isa(λ[1],PDScalF) || error("1×1 λ section should be a PDScalF type")
        lam = λ[1].s
        lamsq = lam*lam
        for j in 1:ℓ₁
            sc = lam/(s.L₁₁[1,j] = sqrt(s.A₁₁[1,j]*lamsq + 1.))
            scale!(view(L₂₁,:,j),sc)
            scale!(view(L₃₁,:,j),sc)
        end
    else
        Ac_mul_B!(λ₁,copy!(s.L₁₁,s.A₁₁)) # multiply on left by λ₁'
        for i₁ in inds(p₁,ℓ₁)
            wL = A_mul_B!(view(s.L₁₁,:,i₁),λ₁)
            for j in 1:p₁               # inflate diagonal
                wL[j,j] += 1.0
                for i in 1:(j-1)        # zero lower triangle (cosmetic)
                    wL[i,j] = 0.
                end
            end
            _,info = LAPACK.potrf!('L',wL) # lower Cholesky factor
            info == 0 || error("Cholesky failure at L₁₁ block $(k+1)")
            tr = Triangular(wL,:L,false)
            A_rdiv_Bc!(A_mul_B!(view(L₂₁,:,i₁),λ₁),tr)
            A_rdiv_Bc!(A_mul_B!(view(L₃₁,:,i₁),λ₁),tr)
        end
    end
                                        # second level updates
    L₂₂d = fill!(s.L₂₂.data,0.)
    L₃₂ = copy!(s.L₃₂,s.A₃₂)
    if p₂ == 1
        isa(λ₂,PDScalF) || error("1×1 λ section should be a PDScalF type")
        lam = λ₂.s
        for i in 1:ℓ₂
            L₂₂d[i,i] = s.A₂₂[1,i]*lam*lam + 1.
        end
        scale!(L₂₁,lam)
        scale!(L₃₂,lam)
    else
        Ac_mul_B!(λ₂,reshape(L₂₁,(p₂,ℓ₂*q₁)))
        for i₂ in inds(p₂,ℓ₂)
            wL = copy!(view(s.L₂₂.data,i₂,i₂),view(s.A₂₂,:,i₂))
            Ac_mul_B!(λ₂,A_mul_B!(wL,λ₂)) # convert the block to λ₂A₂₂λ₂'
            for j in 1:p₂                 # inflate diagonal
                wL[j,j] += 1.
                for i in 1:(j-1)        # zero the strict upper triangle
                    wL[i,j] = 0.
                end
            end
            A_mul_B!(view(L₃₂,:,i₂),λ₂)
        end
    end
    BLAS.syrk!('L','N',-1.,L₂₁,1.,L₂₂d)
    _, info = LAPACK.potrf!('L',L₂₂d)
    info == 0 ||  error("downdated Z₂'Z₂ is not positive definite")
    A_rdiv_Bc!(BLAS.gemm!('N','T',-1.,L₃₁,L₂₁,1.,L₃₂),s.L₂₂)
    L₃₃d = copy!(s.L₃₃.UL,s.A₃₃)
    BLAS.syrk!('L','N',-1.,L₃₁,1.,BLAS.syrk!('L','N',-1.,L₃₂,1.,L₃₃d))
    _, info = LAPACK.potrf!('L',L₃₃d)
    info == 0 ||  error("downdated X'X is not positive definite")
    s
end

## grad calculation
##  tr((∂λ/∂θᵢ)*(LL')⁻¹*λ'*Z'Z) + tr((∂λ'/∂θᵢ)'*Z'Z*λ*(LL')⁻¹)
## Because (∂λ/∂θᵢ) is block diagonal, only the diagonal blocks of
## (L'L)⁻¹λ'Z'Z must be evaluated and summed.

function grad!(res::Vector{Matrix{Float64}},s::PLSTwo,λ::Vector)
    length(λ) == length(res) == 2 || throw(DimensionMismatch(""))
    _,p₁,p₂,ℓ₁,ℓ₂,q₁,q₂ = size(s)
    λ₁ = λ[1]
    r₁ = res[1]
    size(λ₁) == size(r₁) == (p₁,p₁) || throw(DimensionMismatch(""))
    λ₂ = λ[2]
    r₂ = res[2]
    size(λ₂) == size(r₂) == (p₂,p₂) || throw(DimensionMismatch(""))
    tmp₁ = similar(r₁)
    tmp₃ = Array(Float64,q₂,p₁)
    tmp₄ = Array(Float64,q₂,p₂)
    tmp₅ = Array(Float64,p₁,p₂)
    inds₁ = 1:p₁
    for k₁ in 1:ℓ₁  ## create an iterator for this idiom
        copy!(tmp₁,view(s.A₁₁,:,inds₁))
        L₁ = Triangular(view(s.L₁₁,:,inds₁),:L,false)
        A_ldiv_B!(L₁,A_mul_B!(λ₁,tmp₁))
        copy!(tmp₃,view(s.A₂₁,:,inds₁))
        inds₂ = 1:p₂
        for k₂ in 1:ℓ₂
            t2kk = A_mul_B!(λ₂,view(tmp₃,inds₂,:))
            BLAS.gemm!('N','N',-1.0,view(s.L₂₁,inds₂,inds₁),tmp₁,1.0,t2kk)
            inds₂ += p₂
        end
        A_ldiv_B!(cholesky(s.L₂₂),tmp₃)
        Ac_ldiv_B!(L₁,BLAS.gemm!('T','N',-1.0,view(s.L₂₁,:,inds₁),tmp₃,1.0,tmp₁))
        for j in 1:p₁, i in j:p₁
            r₁[i,j] += tmp₁[i,j]
        end
        inds₁ += p₁
    end
    inds₂ = 1:p₂
    for k₂ in 1:ℓ₂
        fill!(tmp₄,0.)
        dd = view(tmp₄,inds₂,:)         # diagonal block for k₂
        A_mul_B!(λ₂,copy!(view(tmp₄,inds₂,:),view(s.A₂₂,:,inds₂)))
        inds₁ = 1:p₁
        for k₁ in 1:ℓ₁
            A_ldiv_B!(Triangular(view(s.L₁₁,:,inds₁),:L,false),
                      A_mul_B!(λ₁,Base.LinAlg.transpose!(tmp₅,view(s.A₂₁,inds₂,inds₁))))
            inds₃ = 1:p₂
            for k₃ in 1:ℓ₂
                BLAS.gemm!('N','N',-1.0,view(s.L₂₁,inds₃,inds₁),tmp₅,1.0,view(tmp₄,inds₃,:))
                inds₃ += p₂
            end
            inds₁ += p₁
        end
        A_ldiv_B!(cholesky(s.L₂₂),tmp₄)
        for j in 1:p₂, i in j:p₂
            r₂[i,j] += dd[i,j]
        end
        inds₂ += p₂
    end
    res
end

function Base.full(s::PLSTwo)
    _,p₁,p₂,ℓ₁,ℓ₂,q₁,q₂ = size(s)
    ntot = q₁ + q₂
    A = zeros(ntot,ntot)
    L = zeros(ntot,ntot)
    for i₁ in inds(p₁,ℓ₁)
        copy!(view(A,i₁,i₁),view(s.A₁₁,:,i₁))
        copy!(view(L,i₁,i₁),view(s.L₁₁,:,i₁))
    end
    i₂₁ = q₁+(1:q₂)
    copy!(view(A,i₂₁,:),s.A₂₁)
    copy!(view(L,i₂₁,:),s.L₂₁)
    A₂₂ = view(A,i₂₁,i₂₁)
    for i₂ in inds(p₂,ℓ₂)
        copy!(view(A₂₂,i₂,i₂),view(s.A₂₂,:,inds₂))
    end
    copy!(view(L,inds₂₁,inds₂₁),s.L₂₂)
    Base.LinAlg.copytri!(A,'L'), cholesky(L,:L)
end

function Base.full(s::PLSTwo,λ::Vector)
    length(λ) == 2 || throw(DimensionMismatch(""))
    _,p₁,p₂,ℓ₁,ℓ₂,q₁,q₂ = size(s)
    ntot = q₁ + q₂
    res = zeros(ntot,ntot)
    λ₁ = tril(λ[1])
    λ₂ = tril(λ[2])
    for i₁ in inds(p₁,ℓ₁)
        copy!(view(res,i₁,i₁),λ₁)
    end
    for i₂ in 1:ℓ₂
        copy!(view(res,q₁+i₂,q₁+i₂),λ₂)
    end
    res
end
