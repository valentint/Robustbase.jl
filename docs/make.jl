using Documenter
using Robustbase

##  DocMeta.setdocmeta!(Robustbase, :DocTestSetup, :(using Robustbase); recursive=true)

makedocs(sitename = "Robustbase",
    format = Documenter.HTML(),
    modules = [Robustbase],
    warnonly = [:missing_docs, :doctest]
    ## warnonly = [:missing_docs, :docs_block]
)

deploydocs(
    repo = "github.com/valentint/Robustbase.jl.git",
    target = "build",
    deps   = nothing,
    make   = nothing,
    push_preview = true,
)
