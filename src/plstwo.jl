type PLSTwo <: PLSSolver   # Solver for models with two crossed or nearly crossed terms
    Ad::Array{Float64,3}                # diagonal blocks
    Ab::Array{Float64,3}                # base blocks
    At::Symmetric{Float64}              # lower right block
    Ld::Array{Float64,3}                # diagonal blocks
    Lb::Array{Float64,3}                # base blocks
    Lt::Base.LinAlg.Cholesky{Float64}   # lower right triangle
    Zt::SparseMatrixCSC
end

function PLSTwo(facs::Vector,Xst::Vector,Xt::Matrix)
    length(facs) == length(Xst) == 2 || throw(DimensionMismatch("PLSTwo"))
    nl = [length(f.pool) for f in facs]
    n = [size(x,1) for x in Xst]
    dblksz = nl[2] * n[2]
    nl[1] * n[1] >= dblksz || error("reverse the order of the random effects terms")
    dblksz += size(Xt,1)
    Ad = zeros(n[1],n[1],nl[1])
    Ab = zeros(dblksz,n[1],nl[1])
end

    
    
