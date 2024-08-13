module IssueData
    using Arrow
    using CSV
    using DataFrames
    using Downloads
    using Scratch
    using ZipFile

    export get_data

    const CACHE = Ref("")
    const URL = "https://github.com/user-attachments/files/16604579/testdataforjulia_bothcase.zip"

    function extract_csv(zipfile, fname; delim=',', header=1, kwargs...)
        file = only(filter(f -> endswith(f.name, fname), zipfile.files))
        return CSV.read(file, DataFrame; delim, header, kwargs...)
    end

    function get_data()
        path = joinpath(CACHE[], "780.arrow")

        isfile(path) && return DataFrame(Arrow.Table(path); copycols=true)

        @info "downloading..."
        data = open(Downloads.download(URL), "r") do io
            zipfile = ZipFile.Reader(io)
            @info "extracting..."
            return extract_csv(zipfile, "testdataforjulia_bothcase.csv")
        end

        Arrow.write(path, data)
        return data
    end

    clear_scratchspaces!() = Scratch.clear_scratchspaces!(@__MODULE__)

    function __init__()
        CACHE[] = @get_scratch!("data")
        return nothing
    end
end

using .IssueData
using MixedModels

data = get_data()
m0form = @formula(case ~ 0 + Analysisclass + (1|cropyear/individual_local_identifier))
# works
model_fast = fit(MixedModel, m0form, data, Bernoulli();
            wts=float.(data.weights),
            contrasts= Dict(:Analysisclass => DummyCoding(; base="aRice_Wet_day")),
            fast=true,
            progress=true,
            verbose=false)
# fails
model = fit(MixedModel, m0form, data, Bernoulli();
            wts=float.(data.weights),
            contrasts= Dict(:Analysisclass => DummyCoding(; base="aRice_Wet_day")),
            fast=false,
            progress=true,
            verbose=false)

# works        
model = fit(MixedModel, m0form, data, Bernoulli();
            wts=float.(data.weights),
            contrasts= Dict(:Analysisclass => DummyCoding(; base="aRice_Wet_day")),
            init_from_lmm=[:θ],
            fast=false,
            progress=true,
            verbose=false)
