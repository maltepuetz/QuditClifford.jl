using QuditClifford
using Documenter
using DocumenterVitepress

DocMeta.setdocmeta!(QuditClifford, :DocTestSetup, :(using QuditClifford); recursive=true)

makedocs(;
    modules=[QuditClifford],
    authors="Malte Pütz",
    sitename="QuditClifford.jl",
    checkdocs=:exports,
    format=DocumenterVitepress.MarkdownVitepress(;
        repo="github.com/maltepuetz/QuditClifford.jl",
        devbranch="main",
        devurl="dev",
        deploy_url="https://maltepuetz.github.io/QuditClifford.jl",
    ),
    pages=[
        "Home" => "index.md",
        "Manual" => [
            "Getting Started" => "getting-started.md",
            "Examples" => "examples.md",
            "Representation and Conventions" => "conventions.md",
            "Measurements and Expectations" => "measurements.md",
        ],
        "API Reference" => "api.md",
    ],
)

DocumenterVitepress.deploydocs(;
    repo="github.com/maltepuetz/QuditClifford.jl",
    target=joinpath(@__DIR__, "build"),
    branch="gh-pages",
    devbranch="main",
    push_preview=true,
)
