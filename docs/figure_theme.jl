# Light/dark display of the CairoMakie figures in the docs, included by make.jl. The figure
# style itself is in makie_theme.jl. Every Figure that an @example block returns is rendered
# twice as SVG, with a transparent figure background: once as it is, for the light theme, and
# once restyled for the dark theme (see `_dark_svg`). SVG keeps lines and text sharp at any
# zoom level and pixel density, where a PNG of fixed resolution turns blurry.
#
# The two SVGs are written next to the page as `<name>_light.svg{.light-only}` and
# `<name>_dark.svg{.dark-only}` by the MIME type FIGURE_MIME below, which `display_dict` produces
# for figures. It follows DocumenterVitepress's "image/png+lightdark", which Documenter never
# produces itself, but carries the figure size, so that the image is shown at the size of the
# figure in CSS pixels. The CSS rule in src/.vitepress/config.mts shows the image that matches
# the VitePress theme toggle.

using CairoMakie
using Documenter
using DocumenterVitepress
using CairoMakie.Makie.Colors: Colorant, Lab, RGB, hex

# colors of the VitePress dark theme
const DARK_TEXT = RGBf(0.875, 0.875, 0.84)       # --vp-c-text-1
const DARK_PANEL = RGBf(0.125, 0.129, 0.153)     # --vp-c-bg-soft, background of axes and legends
const DARK_FRAME = RGBf(0.45, 0.45, 0.47)        # spines, ticks and legend frames

# The dark variant maps every color in CIELAB. Neutral colors (black, grays, white) flip their
# lightness, so black lines turn light and white markers dark. Chromatic colors keep hue and
# chroma, and their lightness is compressed into a band that is visible on the dark panel; this
# keeps the order of sequential colors (e.g. light to dark blue for increasing L), so legends
# read the same in both themes. Colormaps flip their lightness, so that low values recede into
# the dark background, as usual for dark themes.
_neutral(lab) = hypot(lab.a, lab.b) < 12
function _dark_color(c)
    rgba = Makie.to_color(c)
    lab = convert(Lab, RGB(rgba.r, rgba.g, rgba.b))
    l = _neutral(lab) ? 100 - 0.85 * lab.l : 35 + 0.6 * lab.l
    rgb = convert(RGB, Lab(l, lab.a, lab.b))
    return RGBAf(clamp(rgb.r, 0, 1), clamp(rgb.g, 0, 1), clamp(rgb.b, 0, 1), rgba.alpha)
end
function _dark_colormap_color(c)
    lab = convert(Lab, RGB(c.r, c.g, c.b))
    rgb = convert(RGB, Lab(100 - lab.l, lab.a, lab.b))
    return RGBAf(clamp(rgb.r, 0, 1), clamp(rgb.g, 0, 1), clamp(rgb.b, 0, 1), c.alpha)
end
# Plot attributes keep the type of the value they were created with, e.g. a Symbol for
# `color=:black`, so the dark value is given in the same type.
function _dark_input(v, resolved=v)
    if v isa Union{Symbol,AbstractString}
        # placeholders such as :cycled (a color of the palette) take the resolved color
        c = Makie.to_color(v === :cycled ? resolved : v)
        iszero(c.alpha) && return v  # :transparent
        h = "#" * hex(RGB(_dark_color(c)))
        return v isa Symbol ? Symbol(h) : h
    elseif v isa Colorant
        return convert(typeof(v), _dark_color(v))
    elseif v isa AbstractVector{<:Colorant}
        return convert.(eltype(v), _dark_color.(v))
    elseif v isa Tuple{Any,Real}  # (color, alpha)
        return (_dark_input(v[1]), v[2])
    end
    return v  # numbers mapped through a colormap, or no color
end

# Colormaps given by name (or as Reverse(name)) are replaced by the name of a registered
# colormap with flipped lightness.
const DARK_COLORMAP_NAMES = Dict{Vector{RGBAf},Symbol}()
function _register_colormap(cmap)
    get!(DARK_COLORMAP_NAMES, cmap) do
        name = Symbol("docs_dark_", length(DARK_COLORMAP_NAMES) + 1)
        CS = Makie.PlotUtils.ColorSchemes
        CS.colorschemes[name] = CS.ColorScheme(cmap)
        push!(Makie.all_gradient_names, string(name))
        name
    end
end
function _dark_colormap_input(v)
    cmap = _dark_colormap_color.(Makie.to_colormap(v))
    v isa AbstractVector{<:Colorant} && return convert.(eltype(v), cmap)
    v isa Reverse && return Reverse(_register_colormap(reverse(cmap)))
    name = _register_colormap(cmap)
    return v isa AbstractString ? string(name) : name
end

# (block type, attribute => dark value) for the blocks of a figure
const DARK_BLOCK_ATTRIBUTES = (
    (Axis, (:backgroundcolor => DARK_PANEL,
        (a => DARK_TEXT for a in (:titlecolor, :subtitlecolor, :xlabelcolor, :ylabelcolor,
            :xticklabelcolor, :yticklabelcolor))...,
        (a => DARK_FRAME for a in (:leftspinecolor, :rightspinecolor, :topspinecolor,
            :bottomspinecolor, :xtickcolor, :ytickcolor, :xminortickcolor,
            :yminortickcolor))...)),
    (Legend, (:backgroundcolor => DARK_PANEL, :framecolor => DARK_FRAME, :labelcolor => DARK_TEXT,
        :titlecolor => DARK_TEXT)),
    (Colorbar, (:labelcolor => DARK_TEXT, :ticklabelcolor => DARK_TEXT,
        (a => DARK_FRAME for a in (:leftspinecolor, :rightspinecolor, :topspinecolor,
            :bottomspinecolor, :tickcolor, :minortickcolor))...)),
    (Label, (:color => DARK_TEXT,)),
)

# CairoMakie gives the SVG the figure size as its width and height, so one figure unit is one
# CSS pixel.
function _svg(fig)
    io = IOBuffer()
    show(io, MIME"image/svg+xml"(), fig)
    return take!(io)
end

# the value an attribute was set to (for plots, before any conversion)
_input(x::Makie.Block, name) = getproperty(x, name)[]
_input(x, name) = x.attributes.inputs[name].value

# set the attribute `name` of `x` (a block or a plot) to `value`, remembering the old value
function _restyle!(changed, x, name, value)
    old = _input(x, name)
    setproperty!(x, name, value)
    push!(changed, (x, name, old))
    return nothing
end

# render fig in the dark style, then restore it
function _dark_svg(fig)
    changed = []
    try
        # colorbars that follow the colormap of a plot change with it and must not be
        # restyled a second time
        colorbars = [cb => Makie.to_colormap(cb.colormap[]) for cb in fig.content if cb isa Colorbar]
        for block in fig.content
            for (T, attributes) in DARK_BLOCK_ATTRIBUTES
                block isa T || continue
                for (name, value) in attributes
                    _restyle!(changed, block, name, value)
                end
            end
            block isa Axis || continue
            for plot in block.scene.plots
                for name in (:color, :strokecolor)
                    haskey(plot, name) || continue
                    _restyle!(changed, plot, name, _dark_input(_input(plot, name), plot[name][]))
                end
                haskey(plot, :colormap) || continue
                _restyle!(changed, plot, :colormap, _dark_colormap_input(_input(plot, :colormap)))
            end
        end
        for (cb, cmap) in colorbars
            Makie.to_colormap(cb.colormap[]) == cmap || continue
            _restyle!(changed, cb, :colormap, _dark_colormap_input(_input(cb, :colormap)))
        end
        return _svg(fig)
    finally
        for (x, name, old) in reverse(changed)
            setproperty!(x, name, old)
        end
    end
end

const FIGURE_MIME = MIME"image/svg+lightdark+sized"
DocumenterVitepress.mime_priority(::FIGURE_MIME) = 7.5

function DocumenterVitepress.render_mime(io::IO, ::FIGURE_MIME, node, (light, dark, (width, height)),
    page, doc; md_output_path, kwargs...)
    name = String(rand('a':'z', 7))
    dir = joinpath(doc.user.build, md_output_path, dirname(relpath(page.build, doc.user.build)))
    for (variant, svg) in (("light", light), ("dark", dark))
        write(joinpath(dir, "$(name)_$(variant).svg"), svg)
        println(io, "![]($(name)_$(variant).svg){.$(variant)-only width=$(width)px ",
            "height=$(height)px}")
    end
end

# Errorbars and rangebars draw their caps as scatter markers, which take the stroke of the
# Scatter theme in makie_theme.jl. A cap belongs to its bar, so it is drawn without one.
_caps(fig) = [cap for block in fig.content if block isa Axis
              for plot in block.scene.plots if plot isa Union{Errorbars,Rangebars}
              for cap in plot.plots if cap isa Scatter]

function Documenter.display_dict(fig::Figure; context=nothing)
    background = fig.scene.backgroundcolor[]
    fig.scene.backgroundcolor[] = RGBAf(0, 0, 0, 0)
    changed = []
    light, dark = try
        for cap in _caps(fig)
            _restyle!(changed, cap, :strokewidth, 0)
        end
        _svg(fig), _dark_svg(fig)
    finally
        for (x, name, old) in reverse(changed)
            setproperty!(x, name, old)
        end
        fig.scene.backgroundcolor[] = background
    end
    width, height = round.(Int, size(fig.scene))
    return Dict{MIME,Any}(FIGURE_MIME() => (light, dark, (width, height)))
end
