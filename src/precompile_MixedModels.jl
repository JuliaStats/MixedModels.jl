function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    let fbody = try Base.bodyfunction(which(fit!, (LinearMixedModel,))) catch missing end
    if !ismissing(fbody)
        precompile(fbody, (Bool,Bool,Nothing,Int64,typeof(fit!),LinearMixedModel,))
    end
end   # time: 6.582904
    Base.precompile(Tuple{typeof(fit!),GeneralizedLinearMixedModel})   # time: 6.0995073
    Base.precompile(Tuple{Type{LinearMixedModel},AbstractArray,Tuple,FormulaTerm,Vector{Any},Nothing})   # time: 2.2513552
    let fbody = try Base.bodyfunction(which(fit!, (LinearMixedModel{Float64},))) catch missing end
    if !ismissing(fbody)
        precompile(fbody, (Bool,Bool,Nothing,Int64,typeof(fit!),LinearMixedModel{Float64},))
    end
end   # time: 2.1690955
    isdefined(MixedModels, Symbol("#22#23")) && Base.precompile(Tuple{getfield(MixedModels, Symbol("#22#23"))})   # time: 0.70335937
    Base.precompile(Tuple{Type{LinearMixedModel},Vector{Float64},Tuple{Matrix{Float64}, ReMat{Float64, 2}},FormulaTerm{ContinuousTerm{Float64}, Tuple{MatrixTerm{Tuple{InterceptTerm{true}, ContinuousTerm{Float64}}}, RandomEffectsTerm}},Vector{Any},Nothing})   # time: 0.6737979
    Base.precompile(Tuple{typeof(deviance!),GeneralizedLinearMixedModel,Int64})   # time: 0.6433022
    Base.precompile(Tuple{Type{LinearMixedModel},Vector{Float64},FeTerm{Float64, Matrix{Float64}},Vector{AbstractReMat{Float64}},FormulaTerm{ContinuousTerm{Float64}, Tuple{MatrixTerm{Tuple{InterceptTerm{true}, ContinuousTerm{Float64}}}, RandomEffectsTerm}},Vector{Any},Nothing})   # time: 0.42538697
    Base.precompile(Tuple{typeof(deviance!),GeneralizedLinearMixedModel{Float64, Bernoulli{Float64}},Int64})   # time: 0.3716886
    Base.precompile(Tuple{typeof(_ranef_refs),InteractionTerm{Tuple{CategoricalTerm{StatsModels.FullDummyCoding, String, 2}, CategoricalTerm{StatsModels.FullDummyCoding, String, 60}}},NamedTuple{(:use, :age, :urban, :livch, :dist), Tuple{Vector{String}, Vector{Float64}, Vector{String}, Vector{String}, Vector{String}}}})   # time: 0.15686166
    Base.precompile(Tuple{typeof(fit!),GeneralizedLinearMixedModel{Float64, Bernoulli{Float64}}})   # time: 0.1559177
    Base.precompile(Tuple{typeof(modelcols),RandomEffectsTerm,NamedTuple{(:reaction, :days, :subj), Tuple{Vector{Float64}, Vector{Int8}, Vector{String}}}})   # time: 0.14000398
    Base.precompile(Tuple{typeof(adjA),Vector{Int32},Matrix{Float64}})   # time: 0.11642844
    Base.precompile(Tuple{typeof(apply_schema),Term,MultiSchema{FullRank},UnionAll})   # time: 0.0992202
    Base.precompile(Tuple{typeof(fit),Type{GeneralizedLinearMixedModel},FormulaTerm,NamedTuple,Bernoulli,LogitLink})   # time: 0.09839547
    Base.precompile(Tuple{typeof(scaleinflate!),UniformBlockDiagonal{Float64},ReMat{Float64, 2}})   # time: 0.07856259
    Base.precompile(Tuple{typeof(*),Adjoint{Float64, ReMat{Float64, 2}},ReMat{Float64, 2}})   # time: 0.064247385
    let fbody = try Base.bodyfunction(which(fit, (Type{LinearMixedModel},FormulaTerm,NamedTuple,))) catch missing end
    if !ismissing(fbody)
        precompile(fbody, (Any,Any,Bool,Bool,Any,Any,typeof(fit),Type{LinearMixedModel},FormulaTerm,NamedTuple,))
    end
end   # time: 0.0641711
    Base.precompile(Tuple{typeof(apply_schema),Tuple{RandomEffectsTerm, RandomEffectsTerm},MultiSchema{FullRank},Type{LinearMixedModel}})   # time: 0.062291622
    Base.precompile(Tuple{typeof(apply_schema),RandomEffectsTerm,MultiSchema{FullRank},Type{<:MixedModel}})   # time: 0.05415768
    Base.precompile(Tuple{typeof(_ranef_refs),CategoricalTerm{DummyCoding, String, 17},NamedTuple{(:reaction, :days, :subj), Tuple{Vector{Float64}, Vector{Int8}, Vector{String}}}})   # time: 0.042897593
    Base.precompile(Tuple{typeof(rmulΛ!),Matrix{Float64},ReMat{Float64, 2}})   # time: 0.03744466
    Base.precompile(Tuple{typeof(rdiv!),Matrix{Float64},UpperTriangular{Float64, Adjoint{Float64, UniformBlockDiagonal{Float64}}}})   # time: 0.037409946
    Base.precompile(Tuple{typeof(LD),UniformBlockDiagonal{Float64}})   # time: 0.029826526
    Base.precompile(Tuple{typeof(*),Adjoint{Float64, ReMat{Float64, 1}},ReMat{Float64, 1}})   # time: 0.02977296
    Base.precompile(Tuple{typeof(GLM.wrkresp!),SubArray{Float64, 1, Matrix{Float64}, Tuple{Base.Slice{Base.OneTo{Int64}}, Int64}, true},GLM.GlmResp{Vector{Float64}, Bernoulli{Float64}, LogitLink}})   # time: 0.022571307
    Base.precompile(Tuple{typeof(modelcols),RandomEffectsTerm,NamedTuple{(:use, :age, :urban, :livch, :dist), Tuple{Vector{String}, Vector{Float64}, Vector{String}, Vector{String}, Vector{String}}}})   # time: 0.01701472
    Base.precompile(Tuple{typeof(scaleinflate!),Diagonal{Float64, Vector{Float64}},ReMat{Float64, 1}})   # time: 0.014116756
    Base.precompile(Tuple{typeof(cholUnblocked!),UniformBlockDiagonal{Float64},Type{Val{:L}}})   # time: 0.011265205
    Base.precompile(Tuple{Core.Type{MixedModels.GeneralizedLinearMixedModel{Float64, Distributions.Bernoulli}},LinearMixedModel,Any,Any,Vector,Any,Any,Any,Any,Any,Any,Any,Any,Any,Any})   # time: 0.010291444
    Base.precompile(Tuple{typeof(apply_schema),RandomEffectsTerm,MultiSchema{FullRank},Type{LinearMixedModel}})   # time: 0.009160354
    Base.precompile(Tuple{Type{FeTerm},Matrix{Float64},Any})   # time: 0.008526229
    Base.precompile(Tuple{typeof(apply_schema),Term,MultiSchema{FullRank},Type})   # time: 0.008067499
    Base.precompile(Tuple{typeof(apply_schema),Tuple{RandomEffectsTerm, RandomEffectsTerm},MultiSchema,Type})   # time: 0.006452216
    Base.precompile(Tuple{typeof(*),Adjoint{Float64, FeMat{Float64, Matrix{Float64}}},ReMat{Float64, 2}})   # time: 0.005299216
    Base.precompile(Tuple{typeof(reweight!),ReMat{Float64, 1},Vector{Float64}})   # time: 0.00500359
    Base.precompile(Tuple{typeof(reweight!),ReMat{Float64, 2},Vector{Float64}})   # time: 0.004945764
    Base.precompile(Tuple{typeof(_pushALblock!),Vector{AbstractMatrix{Float64}},Vector{AbstractMatrix{Float64}},Diagonal{Float64, Vector{Float64}}})   # time: 0.004846841
    Base.precompile(Tuple{typeof(apply_schema),Tuple{RandomEffectsTerm, RandomEffectsTerm},MultiSchema{FullRank},Type{<:MixedModel}})   # time: 0.004639013
    Base.precompile(Tuple{typeof(_pushALblock!),Vector{AbstractMatrix{Float64}},Vector{AbstractMatrix{Float64}},UniformBlockDiagonal{Float64}})   # time: 0.004212355
    isdefined(MixedModels, Symbol("#obj#92")) && Base.precompile(Tuple{getfield(MixedModels, Symbol("#obj#92")),Vector{Float64},Vector{Float64}})   # time: 0.00415707
    Base.precompile(Tuple{typeof(reweight!),LinearMixedModel{Float64},Vector{Float64}})   # time: 0.003799728
    Base.precompile(Tuple{typeof(copyto!),UniformBlockDiagonal{Float64},UniformBlockDiagonal{Float64}})   # time: 0.003437127
    Base.precompile(Tuple{typeof(cholUnblocked!),Matrix{Float64},Type{Val{:L}}})   # time: 0.002697967
    isdefined(MixedModels, Symbol("#97#99")) && Base.precompile(Tuple{getfield(MixedModels, Symbol("#97#99")),ReMat{Float64, 1}})   # time: 0.002630397
    Base.precompile(Tuple{typeof(apply_schema),ConstantTerm{Int64},MultiSchema{FullRank},UnionAll})   # time: 0.002451962
    Base.precompile(Tuple{typeof(apply_schema),ConstantTerm{Int64},MultiSchema{FullRank},Type})   # time: 0.002029733
    Base.precompile(Tuple{typeof(rmulΛ!),Matrix{Float64},ReMat{Float64, 1}})   # time: 0.00186467
    Base.precompile(Tuple{typeof(getproperty),LinearMixedModel,Symbol})   # time: 0.001859925
    isdefined(MixedModels, Symbol("#59#61")) && Base.precompile(Tuple{getfield(MixedModels, Symbol("#59#61")),ReMat{Float64, 1}})   # time: 0.00181014
    isdefined(MixedModels, Symbol("#58#60")) && Base.precompile(Tuple{getfield(MixedModels, Symbol("#58#60")),ReMat{Float64, 2}})   # time: 0.001746027
    Base.precompile(Tuple{typeof(mul!),Vector{Float64},ReMat{Float64, 1},Vector{Float64},Float64,Float64})   # time: 0.001642233
    Base.precompile(Tuple{typeof(cholUnblocked!),Diagonal{Float64, Vector{Float64}},Type{Val{:L}}})   # time: 0.001591365
    isdefined(MixedModels, Symbol("#59#61")) && Base.precompile(Tuple{getfield(MixedModels, Symbol("#59#61")),ReMat{Float64, 2}})   # time: 0.001566984
    Base.precompile(Tuple{Type{RandomEffectsTerm},Any,Any})   # time: 0.001504504
    isdefined(MixedModels, Symbol("#58#60")) && Base.precompile(Tuple{getfield(MixedModels, Symbol("#58#60")),ReMat{Float64, 1}})   # time: 0.001407624
    Base.precompile(Tuple{typeof(*),Adjoint{Float64, FeMat{Float64, Matrix{Float64}}},ReMat{Float64, 1}})   # time: 0.001186352
    Base.precompile(Tuple{Type{MultiSchema},FullRank})   # time: 0.001066192
    Base.precompile(Tuple{Type{ReMat{Float64, 2}},CategoricalTerm{DummyCoding, String, 17},Vector{Int32},Vector{String},Vector{String},Matrix{Float64},Matrix{Float64},LowerTriangular{Float64, Matrix{Float64}},Vector{Int64},SparseMatrixCSC{Float64, Int32},Matrix{Float64}})   # time: 0.001038076
end
