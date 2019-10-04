using RCall, MixedModels, Test
const LMM = LinearMixedModel
const GLMM = GeneralizedLinearMixedModel

@testset "RCall for lme4" begin
    reval("""
    if(!require(lme4)){
        # use tmp for tests if lme4 isn't available
        .libPaths("/tmp")
        lib <- .libPaths()[1L]
        install.packages("lme4",repos="https://cloud.r-project.org", libs=lib)
        library(lme4)
    }""")

    @testset "lmerMod" begin
        sleepstudy = rcopy(R"sleepstudy")
        ### from R ###
        jlmm = fit!(LMM(@formula(Reaction ~ 1 + Days + (1 + Days|Subject)),sleepstudy), REML=false)
        rlmm = rcopy(R"m <- lmer(Reaction ~ 1 + Days + (1 + Days|Subject),sleepstudy,REML=FALSE)")

        @test jlmm.θ ≈ rlmm.θ atol=0.001
        @test objective(jlmm) ≈ objective(rlmm) atol=0.001
        @test fixef(jlmm) ≈ fixef(rlmm) atol=0.001

        jlmm = fit!(jlmm, REML=true)
        rlmm = rcopy(R"update(m, REML=TRUE)")

        @test jlmm.θ ≈ rlmm.θ atol=0.001
        @test objective(jlmm) ≈ objective(rlmm) atol=0.001
        @test fixef(jlmm) ≈ fixef(rlmm) atol=0.001

        ### from Julia ###
        fit!(jlmm, REML=true)
        jm = Tuple([jlmm, sleepstudy])
        @rput jm
        @test rcopy(R"fitted(jm)") ≈ fitted(jlmm)
        @test rcopy(R"REMLcrit(jm)") ≈ objective(jlmm)

        fit!(jlmm, REML=false)
        jm = Tuple([jlmm, sleepstudy])
        @rput jm
        @test rcopy(R"fitted(jm)") ≈ fitted(jlmm)
        @test rcopy(R"deviance(jm)") ≈ objective(jlmm)

    end
end
