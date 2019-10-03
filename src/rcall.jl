import RCall: rcopy, RClass, rcopytype, reval, S4Sxp, sexp, protect, unprotect, sexpclass, @rput, @rget
# if RCall is available, then so is DataFrames
import DataFrames: DataFrame
# from R
# note that weights are not extracted
function rcopy(::Type{LinearMixedModel}, s::Ptr{S4Sxp})
    f = rcopy(s[:call][:formula])
    data = rcopy(s[:frame])
    θ = rcopy(s[:theta])
    reml = rcopy(s[:devcomp][:dims][:REML]) ≠ 0

    m = LinearMixedModel(f,data)
    m.optsum.REML = reml
    m.optsum.feval = rcopy(s[:optinfo][:feval])
    m.optsum.final = rcopy(s[:optinfo][:val])
    m.optsum.optimizer = Symbol("$(rcopy(s[:optinfo][:optimizer])) (lme4)")
    m.optsum.returnvalue = Bool(rcopy(s[:optinfo][:conv][:opt])) ? :FAILURE : :SUCCESS
    m.optsum.fmin = reml ? rcopy(s[:devcomp][:cmp][:REML]) : rcopy(s[:devcomp][:cmp][:dev])
    updateL!(setθ!(m, θ))
end

rcopytype(::Type{RClass{:lmerMod}}, s::Ptr{S4Sxp}) = LinearMixedModel

# TODO: add alternative methods for tbl
# TODO: fix some conversions -- Julia->R->Julia roundtrip currently due to
#        ERROR: REvalError: Error in function (x, value, pos = -1, envir = as.environment(pos), inherits = FALSE,  :
#          SET_VECTOR_ELT() can only be applied to a 'list', not a 'character'
function sexp(::Type{RClass{:lmerMod}}, x::Tuple{LinearMixedModel{T}, DataFrame}) where T
    m, tbl = x
    # should we assume the user is smart enough?
    reval("library(lme4)")

    data = tbl
    formula = String(Symbol(m.formula))
    θ = m.θ

    REML = m.optsum.REML ? "TRUE" : "FALSE"
    par = m.optsum.final
    fval = m.optsum.fmin
    feval = m.optsum.feval
    conv = m.optsum.returnvalue == :SUCCESS ? 0 : 1
    optimizer = String(m.optsum.optimizer)
    message = "fit with MixedModels.jl"
    # yes, it overwrites any variable named data, but you shouldn't be naming
    # your variables that anyway!
    @rput data
    @rput par

    r = """
         parsedFormula <- lFormula(formula=$(formula),
                                   data=data,
                                   REML=$(REML))
         # this bit should probably be reworked to extract the julia fields
         # but it's easier to just let lme4 do a single step and the internal
         # representations are slightly different anyway
         devianceFunction <- do.call(mkLmerDevfun, parsedFormula)
         optimizerOutput <- optimizeLmer(devianceFunction,start=par,
                                         control=list(maxeval=1,calc.derivs=FALSE))
         optimizerOutput\$feval <- $(feval)
         optimizerOutput\$message <- "$(message)"
         optimizerOutput\$optimizer <- "$(optimizer)"

         rho <- environment(devianceFunction)

         mkMerMod(rho = rho,
                 opt = optimizerOutput,
                 reTrms = parsedFormula\$reTrms,
                 fr = parsedFormula\$fr)
    """
    @debug r
    r = reval(r)
    r = protect(sexp(r))
    unprotect(1)
    r
end

sexpclass(x::Tuple{LinearMixedModel{T}, DataFrame}) where T = RClass{:lmerMod}
