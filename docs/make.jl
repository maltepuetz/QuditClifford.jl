using QuditClifford
using Documenter

DocMeta.setdocmeta!(QuditClifford, :DocTestSetup, :(using QuditClifford); recursive=true)

makedocs(;
    modules=[QuditClifford],
    authors="Malte Pütz",
    sitename="QuditClifford.jl",
    format=Documenter.HTML(;
        canonical="https://maltepuetz.github.io/QuditClifford.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/maltepuetz/QuditClifford.jl",
    devbranch="main",
)
