using Weave

function weavit(fnm::AbstractString)
    weave(
        joinpath("docs", "jmd", fnm),
        doctype = "github",
        fig_path = joinpath("docs", "assets"),
        fig_ext = ".svg",
        out_path = joinpath("docs", "src"),
    )
end

#weavit("constructors.jmd")
weavit("optimization.jmd")
#weavit("bootstrap.jmd")
#weavit("SimpleLMM.jmd")
#weavit("MultipleTerms.jmd")
#weavit("SingularCovariance.jmd")
#weavit("SubjectItem.jmd")
#weavit("GaussHermite.jmd")
