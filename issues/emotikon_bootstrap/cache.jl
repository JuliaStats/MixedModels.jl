module Cache
using Downloads
using Scratch
# This will be filled in inside `__init__()`
download_cache = ""
url = "https://github.com/RePsychLing/SMLP2022/raw/main/data/fggk21.arrow"
#"https://github.com/bee8a116-0383-4365-8df7-6c6c8d6c1322"

function data_path()
    fname = joinpath(download_cache, basename(url))
    if !isfile(fname)
        @info "Local cache not found, downloading"
        Downloads.download(url, fname)
    end
    return fname
end

function __init__()
    global download_cache = get_scratch!(@__MODULE__, "downloaded_files")
    return nothing
end
end
