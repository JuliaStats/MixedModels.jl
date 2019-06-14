@deprecate lmm(f::Formula, fr::AbstractDataFrame) LinearMixedModel(f, fr)
@deprecate glmm(f::Formula, fr::AbstractDataFrame, d::Distribution) GeneralizedLinearMixedModel(f, fr, d)
@deprecate glmm(f::Formula, fr::AbstractDataFrame, d::Distribution, l::GLM.Link) GeneralizedLinearMixedModel(f, fr, d, l)
