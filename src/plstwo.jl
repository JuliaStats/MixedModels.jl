type PLSTwo <: PLSSolver   # Solver for models with two crossed or nearly crossed terms
    Ad::Array{Float64,3}                # diagonal blocks
    Ab::Array{Float64,3}                # base blocks
    At::Symmetric{Float64}              # lower right block
    Ld::Array{Float64,3}                # diagonal blocks
    Lb::Array{Float64,3}                # base blocks
    Lt::Base.LinAlg.Cholesky{Float64}   # lower right triangle
    p::Int                              # number of columns in f.e. model matrix
    p2::Int                             # number random effects per level of facs[2]
end

function PLSTwo(facs::Vector,Xst::Vector,Xt::Matrix)
    length(facs) == length(Xst) == 2 || throw(DimensionMismatch("PLSTwo"))
    L = size(Xt,2)
    all(L .== [size(x,2) for x in Xst]) && all(L .== map(length,facs)) || throw(DimensionMismatch(""))
    nl = [length(f.pool) for f in facs]
    pv = [size(x,1) for x in Xst]
    q = nl .* pv
                                        # Do this in lmm instead of checking here
    q[1] >= q[2] || error("reverse the order of the random effects terms")
    p = size(Xt,1)
    p2 = pv[2]
    Ad = zeros(pv[1],pv[1],nl[1])
    Ab = zeros(q[2]+p,pv[1],nl[1])
    finds = q[2] + (1:p)         # indices for the fixed-effects parts
    Abb = view(Ab,finds,:,:)
    At = zeros(q[2]+p,q[2]+p)
    BLAS.syrk!('L','N',1.0,Xt,0.0,view(At,finds,finds))
    r1 = facs[1].refs
    r2 = facs[2].refs
    xst1 = Xst[1]
    xst2 = Xst[2]
    for j in 1:L
        j1 = r1[j]
        inds2 = (r2[j]-1)*p2 + (1:p2)
        BLAS.syr!('L',1.0,view(xst1,:,j),view(Ad,:,:,j1))
        BLAS.syr!('L',1.0,view(xst2,:,j),view(At,inds2,inds2))
        BLAS.ger!(1.0,view(xst2,:,j),view(xst1,:,j),view(Ab,inds2,:,j1))
        BLAS.ger!(1.0,view(Xt,:,j),view(xst1,:,j),view(Abb,:,:,j1))
        BLAS.ger!(1.0,view(Xt,:,j),view(xst2,:,j),view(At,finds,inds2))
    end
    Lt = copy(At)
    for j in 1:q[2]          # inflate diagonal before taking Cholesky
        Lt[j,j] += 1.
    end
    PLSTwo(Ad,Ab,Symmetric(At,:L),similar(Ad),similar(Ab),cholfact!(Lt,:L),p,p2)
end

function Base.cholfact(s::PLSTwo,RX::Bool=true)
    p,p1,p2,l1 = size(s)
    m,n,l = size(s.Lb)
    q2 = m - p
    rem(q2,p2) == 0 || throw(DimensionMisMatch(""))
    Lt = tril!(s.Lt[:L].data)
    RX && (pinds = (q2+1):m; return(Cholesky(Lt[pinds,pinds],'L')))
    L11 = blkdiag({sparse(tril(view(s.Ld,:,:,j))) for j in 1:size(s.Ld,3)}...)
    hcat(vcat(L11,sparse(reshape(s.Lb[1:q2,:,:],(q2,size(L11,2))))),
         vcat(spzeros(size(L11,1),q2),sparse(Lt[1:q2,1:q2])))
end

function Base.logdet(s::PLSTwo,RX=true)
    RX && return(logdet(cholfact(s)))
    p,p1,p2,l1 = size(s)
    m,n,l = size(s.Lb)
    q2 = m - p
    rem(q2,p2) == 0 || throw(DimensionMisMatch(""))
    Lt = s.Lt[:L].data
    sm = 0.
    Ld = s.Ld
    for j in 1:l1, i in 1:p1
        sm += log(Ld[i,i,j])
    end
    for i in 1:q2
        sm += log(Lt[i,i])
    end
    2.sm
end


## arguments u and β contain λ'Z'y and X'y on entry
function plssolve!(s::PLSTwo,u,β)
    p,p1,p2,l = size(s)
    m,n,l = size(s.Ab)
    q2 = m - p
    q1 = p1 * l
    (q = length(u)) == p1*l + q2 || throw(DimensionMismatch(""))

    cu1 = reshape(u[1:q1],(p1,l))
    cu2 = contiguous_view(u,q1,(q2,))
    bb = vcat(cu2,β)
    if n == 1                           # short cut for scalar r.e.
        Linv = [inv(l)::Float64 for l in s.Ld]
        scale!(cu1,Linv)
        LXZ = reshape(s.Lb,(m,l))
        A_ldiv_B!(s.Lt,BLAS.gemv!('N',-1.,LXZ,vec(cu1),1.,bb)) # solve for β
        BLAS.gemv!('T',-1.,LXZ,bb,1.0,vec(cu1))                # cᵤ -= LZX'β
        scale!(cu1,Linv)
    else
        for j in 1:l                    # solve L cᵤ = λ'Z'y blockwise
            BLAS.trsv!('L','N','N',view(s.Ld,:,:,j),view(cu1,:,j))
        end
        LZX = reshape(s.Lb,(m,n*l))
                                        # solve (L_X L_X')̱β = X'y - L_XZ cᵤ
        A_ldiv_B!(s.Lt,BLAS.gemv!('N',-1.0,LZX,vec(cu1),1.0,bb))
                                        # cᵤ := cᵤ - L_XZ'β
        BLAS.gemv!('T',-1.0,LZX,bb,1.0,vec(cu1))
        for j in 1:l                    # solve L'u = cᵤ blockwise
            BLAS.trsv!('L','T','N',view(s.Ld,:,:,j),view(cu1,:,j))
        end
    end
    copy!(view(u,1:q1),cu1)
    copy!(cu2,view(bb,1:q2))
    copy!(β,view(bb,q2+(1:p)))
end

Base.size(s::PLSTwo) = ((m,n,l) = size(s.Ab); (s.p,n,s.p2,l))

function update!(s::PLSTwo,λ::Vector)
    p2 = s.p2
    length(λ) == 2 && abs(dim(λ[2])) == p2 || throw(DimensionMismatch(""))
    updateLdb!(s,λ[1])
                                        # second level updates
    m,n,l = size(s.Lb)
    p,p1,p2,l = size(s)
    q2 = m - s.p
    rem(q2,p2) == 0 || throw(DimensionMismatch(""))
    Lt = s.Lt[:L].data
    if p2 == 1
        isa(λ[2],PDScalF) || error("1×1 λ section should be a PDScalF type")
        lam = λ[2].s.λ
        for k in 1:l, j in 1:p, i in 1:q2
            s.Lb[i,j,k] *= lam
        end
        scale!(lam,view(Lt,:,1:q2))
        scale!(lam,view(Lt,1:q2,:))
    else
        lam = λ[2]
        for k in 1:div(q2,p2)
            ii = (k - 1)*p2 + (1:p2)
            A_mul_B!(lam,reshape(view(s.Lb,ii,:,:),(p2,n*l)))
            A_mul_Bc!(view(Lt,:,ii),lam)
            A_mul_B!(lam,view(Lt,ii,:))
        end
    end
    for j in 1:q2
        Lt[j,j] += 1.
    end
    BLAS.syrk!('L','N',-1.0,reshape(s.Lb,(m,n*l)),1.0,Lt)
    _, info = LAPACK.potrf!('L',Lt)
    info == 0 ||  error("downdated X'X is not positive definite")
    s
end

function updateLdb!(s::Union(PLSOne,PLSTwo),λ::AbstractPDMatFactor)
    k = dim(λ)
    k < 0 || k == size(s.Ad,1) || throw(DimensionMixmatch(""))
    m,n,l = size(s.Ab)
    Lt = copy!(s.Lt.UL,s.At.S)
    Lb = copy!(s.Lb,s.Ab)
    if n == 1                           # shortcut for 1×1 λ
        isa(λ,PDScalF) || error("1×1 λ section should be a PDScalF type")
        lam = λ.s.λ
        lamsq = lam*lam
        Ld = map!(x -> sqrt(x*lamsq + 1.), s.Ld, s.Ad)
        scale!(reshape(Lb,(m,l)),lam ./ vec(Ld))
    else
        Ac_mul_B!(λ,reshape(copy!(s.Ld,s.Ad),(n,n*l)))
        for k in 1:l
            wL = A_mul_B!(view(s.Ld,:,:,k),λ)
            for j in 1:n
                wL[j,j] += 1.0
            end
            _,info = LAPACK.potrf!('L',wL)
            info == 0 || error("Cholesky failure at L diagonal block $k")
            Base.LinAlg.A_rdiv_Bc!(A_mul_B!(view(s.Lb,:,:,k),λ),Triangular(wL,:L,false))
        end
    end
    s
end
