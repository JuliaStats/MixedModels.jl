function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    let fbody = try Base.bodyfunction(which(fit!, (LinearMixedModel,))) catch missing end
    if !ismissing(fbody)
        precompile(fbody, (Bool,Bool,Nothing,Int64,typeof(fit!),LinearMixedModel,))
    end
end   # time: 6.387307
    Base.precompile(Tuple{Type{LinearMixedModel},AbstractArray,Tuple,FormulaTerm,Vector{Any},Nothing})   # time: 2.1642134
    let fbody = try Base.bodyfunction(which(fit!, (LinearMixedModel{Float64},))) catch missing end
    if !ismissing(fbody)
        precompile(fbody, (Bool,Bool,Nothing,Int64,typeof(fit!),LinearMixedModel{Float64},))
    end
end   # time: 1.9979239
    Base.precompile(Tuple{Type{LinearMixedModel},Vector{Float64},Tuple{Matrix{Float64}, ReMat{Float64, 2}},FormulaTerm{ContinuousTerm{Float64}, Tuple{MatrixTerm{Tuple{InterceptTerm{true}, ContinuousTerm{Float64}}}, RandomEffectsTerm}},Vector{Any},Nothing})   # time: 0.6733341
    isdefined(MixedModels, Symbol("#22#23")) && Base.precompile(Tuple{getfield(MixedModels, Symbol("#22#23"))})   # time: 0.64626914
    Base.precompile(Tuple{Type{LinearMixedModel},Vector{Float64},FeTerm{Float64, Matrix{Float64}},Vector{AbstractReMat{Float64}},FormulaTerm{ContinuousTerm{Float64}, Tuple{MatrixTerm{Tuple{InterceptTerm{true}, ContinuousTerm{Float64}}}, RandomEffectsTerm}},Vector{Any},Nothing})   # time: 0.42384765
    Base.precompile(Tuple{typeof(modelcols),RandomEffectsTerm,NamedTuple{(:reaction, :days, :subj), Tuple{Vector{Float64}, Vector{Int8}, Vector{String}}}})   # time: 0.137623
    Base.precompile(Tuple{typeof(adjA),Vector{Int32},Matrix{Float64}})   # time: 0.11764644
    Base.precompile(Tuple{typeof(apply_schema),Term,MultiSchema{FullRank},UnionAll})   # time: 0.07816912
    Base.precompile(Tuple{typeof(scaleinflate!),UniformBlockDiagonal{Float64},ReMat{Float64, 2}})   # time: 0.07240246
    Base.precompile(Tuple{typeof(*),Adjoint{Float64, ReMat{Float64, 2}},ReMat{Float64, 2}})   # time: 0.063550666
    let fbody = try Base.bodyfunction(which(fit, (Type{LinearMixedModel},FormulaTerm,NamedTuple,))) catch missing end
    if !ismissing(fbody)
        precompile(fbody, (Any,Any,Bool,Bool,Any,Any,typeof(fit),Type{LinearMixedModel},FormulaTerm,NamedTuple,))
    end
end   # time: 0.06248636
    Base.precompile(Tuple{typeof(apply_schema),RandomEffectsTerm,MultiSchema{FullRank},Type{<:MixedModel}})   # time: 0.05271745
    Base.precompile(Tuple{typeof(_ranef_refs),CategoricalTerm{DummyCoding, String, 17},NamedTuple{(:reaction, :days, :subj), Tuple{Vector{Float64}, Vector{Int8}, Vector{String}}}})   # time: 0.04189839
    Base.precompile(Tuple{typeof(apply_schema),Tuple{RandomEffectsTerm, RandomEffectsTerm},MultiSchema{FullRank},Type{LinearMixedModel}})   # time: 0.039800957
    Base.precompile(Tuple{typeof(rmulΛ!),Matrix{Float64},ReMat{Float64, 2}})   # time: 0.037355892
    Base.precompile(Tuple{typeof(rdiv!),Matrix{Float64},UpperTriangular{Float64, Adjoint{Float64, UniformBlockDiagonal{Float64}}}})   # time: 0.034880396
    Base.precompile(Tuple{typeof(LD),UniformBlockDiagonal{Float64}})   # time: 0.029555252
    Base.precompile(Tuple{typeof(cholUnblocked!),UniformBlockDiagonal{Float64},Type{Val{:L}}})   # time: 0.011312683
    Base.precompile(Tuple{typeof(apply_schema),RandomEffectsTerm,MultiSchema{FullRank},Type{LinearMixedModel}})   # time: 0.009192527
    Base.precompile(Tuple{typeof(apply_schema),Term,MultiSchema{FullRank},Type})   # time: 0.008316929
    Base.precompile(Tuple{typeof(apply_schema),Tuple{RandomEffectsTerm, RandomEffectsTerm},MultiSchema,Type})   # time: 0.005474164
    Base.precompile(Tuple{typeof(*),Adjoint{Float64, FeMat{Float64, Matrix{Float64}}},ReMat{Float64, 2}})   # time: 0.004983632
    Base.precompile(Tuple{typeof(reweight!),ReMat{Float64, 2},Vector{Float64}})   # time: 0.004720858
    Base.precompile(Tuple{typeof(apply_schema),Tuple{RandomEffectsTerm, RandomEffectsTerm},MultiSchema{FullRank},Type{<:MixedModel}})   # time: 0.004667381
    Base.precompile(Tuple{typeof(_pushALblock!),Vector{AbstractMatrix{Float64}},Vector{AbstractMatrix{Float64}},UniformBlockDiagonal{Float64}})   # time: 0.004252458
    Base.precompile(Tuple{typeof(copyto!),UniformBlockDiagonal{Float64},UniformBlockDiagonal{Float64}})   # time: 0.003408
    Base.precompile(Tuple{typeof(cholUnblocked!),Matrix{Float64},Type{Val{:L}}})   # time: 0.002759723
    Base.precompile(Tuple{typeof(apply_schema),ConstantTerm{Int64},MultiSchema{FullRank},UnionAll})   # time: 0.002100287
    Base.precompile(Tuple{typeof(apply_schema),ConstantTerm{Int64},MultiSchema{FullRank},Type})   # time: 0.002066763
    isdefined(MixedModels, Symbol("#59#61")) && Base.precompile(Tuple{getfield(MixedModels, Symbol("#59#61")),ReMat{Float64, 2}})   # time: 0.00158532
    isdefined(MixedModels, Symbol("#58#60")) && Base.precompile(Tuple{getfield(MixedModels, Symbol("#58#60")),ReMat{Float64, 2}})   # time: 0.001487785
    Base.precompile(Tuple{Type{RandomEffectsTerm},Any,Any})   # time: 0.00135209
    Base.precompile(Tuple{Type{ReMat{Float64, 2}},CategoricalTerm{DummyCoding, String, 17},Vector{Int32},Vector{String},Vector{String},Matrix{Float64},Matrix{Float64},LowerTriangular{Float64, Matrix{Float64}},Vector{Int64},SparseMatrixCSC{Float64, Int32},Matrix{Float64}})   # time: 0.001134242
    Base.precompile(Tuple{Type{MultiSchema},FullRank})   # time: 0.001107174
end
