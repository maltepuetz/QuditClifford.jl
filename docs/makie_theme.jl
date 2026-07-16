# VitePress embeds Makie output at its logical PNG size. Scale every visual
# dimension together so the plots retain their original proportions at 1.5x.
figure_scale = 1.5
scaled(value) = figure_scale * value

size_pt = (450, 270)
darkred = "#BB5566"
qudit_colors = (darkred, "#004488", "#DDAA33", "#228833")

theme = Theme(
    figure_padding=scaled(2),
    size=size_pt,
    fontsize=scaled(8),
    Lines=(
        linewidth=scaled(1.0),
    ),
    Scatter=(
        color=darkred,
        cycle=nothing,
        markersize=scaled(5.0),
        marker=:circle,
        strokewidth=scaled(0.3),
    ),
    Axis=(
        titlefont=:regular,
        titlesize=scaled(10),
        xlabelsize=scaled(10),
        ylabelsize=scaled(10),
        xgridvisible=false,
        ygridvisible=false,
        xticksize=scaled(2.5),
        yticksize=scaled(2.5),
        xminorticksize=scaled(1.5),
        yminorticksize=scaled(1.5),
        spinewidth=scaled(0.75),
        xtickwidth=scaled(0.75),
        ytickwidth=scaled(0.75),
    ),
    Legend=(
        labelfont=:regular,
        padding=(scaled(2), scaled(2), scaled(2), scaled(2)),
        patchlabelgap=scaled(5),
        patchsize=(scaled(9), scaled(4)),
        rowgap=scaled(2),
        colgap=scaled(8),
        titlefont=:regular,
        titlegap=scaled(3),
        margin=(scaled(2), scaled(2), scaled(2), scaled(2)),
        framevisible=false,
    ),
)

theme = merge(theme, theme_latexfonts())
update_theme!(theme)
CairoMakie.activate!(type="svg")
