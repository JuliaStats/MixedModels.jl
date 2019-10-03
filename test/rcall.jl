using RCall, MixedModels, Test
const LMM = LinearMixedModel
const GLMM = GeneralizedLinearMixedModel

@testset "RCall for lme4" begin
    reval("""
    if(!require(lme4)){
        # this is insane, but this part of install.packages to use a userlib
        # because I can't figure out the AppVeyor and Travis configs and otherwise
        # it hangs at the prompt "do you want to create a user....."
        lib <- .libPaths()[1L]
        ok <- dir.exists(lib) & (file.access(lib, 2) == 0L)
        if (length(lib) == 1L && .Platform\$OS.type == "windows") {
            ok <- dir.exists(lib)
            if (ok) {
                fn <- file.path(lib, paste0("_test_dir_", Sys.getpid()))
                unlink(fn, recursive = TRUE)
                res <- try(dir.create(fn, showWarnings = FALSE))
                if (inherits(res, "try-error") || !res)
                    ok <- FALSE
                else unlink(fn, recursive = TRUE)
            }
        }
        if (length(lib) == 1L && !ok) {
            userdir <- unlist(strsplit(Sys.getenv("R_LIBS_USER"), .Platform\$path.sep))[1L]
            lib <- userdir
            if (!file.exists(userdir)) {
                if (!dir.create(userdir, recursive = TRUE)) stop(gettextf("unable to create %s", sQuote(userdir)), domain = NA)
                .libPaths(c(userdir, .libPaths()))
            }

        }

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
