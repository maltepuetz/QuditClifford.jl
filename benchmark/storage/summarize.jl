# Markdown tables from the CSV files of `benchmark/storage/run.jl`:
#
#     julia benchmark/storage/summarize.jl a.csv [b.csv]
#
# For each file:
#   - qudit-major / generator-major per row, per `d` and element type, and the
#     same difference set against one layout switch round trip;
#   - Bits / UInt8 per row at d = 2, per layout.
# With two files, also b / a per row (two machines, or two commits on one).
# Ratios below 1 mean the numerator is faster. Uses only Base and Printf.

using Printf

struct Row
    label::String
    layout::String
    eltype::String
    tableau::String
    d::Int
    n::Int
    ns::Float64
end

function load(path)
    rows = Row[]
    meta = String[]
    for line in eachline(path)
        isempty(line) && continue
        if startswith(line, '#')
            push!(meta, strip(line[2:end]))
            continue
        end
        startswith(line, "label,") && continue
        f = split(line, ',')
        length(f) == 8 || error("$path: expected 8 fields, got $(length(f)): $line")
        push!(rows, Row(f[1], f[2], f[3], f[4], parse(Int, f[5]), parse(Int, f[6]), parse(Float64, f[7])))
    end
    rows, meta
end

key(r::Row) = (r.label, r.layout, r.eltype, r.tableau, r.d, r.n)
fmt(x) = isnan(x) ? "–" : x >= 1e6 ? @sprintf("%.2f ms", x / 1e6) : x >= 1e3 ? @sprintf("%.1f µs", x / 1e3) :
         @sprintf("%.0f ns", x)
ratio(x) = isnan(x) ? "–" : @sprintf("%.2f", x)
uniq(v) = sort!(unique(v))

# rows of the table in order of first appearance
function labels_in(rows, pred)
    out = Tuple{String,String}[]
    for r in rows
        pred(r) || continue
        k = (r.label, r.tableau)
        k in out || push!(out, k)
    end
    out
end

function table(io, title, rows, ns, cell)
    println(io, "\n", title, "\n")
    println(io, "| row | ", join(("n = $n" for n in ns), " | "), " |")
    println(io, "|---|", repeat("---:|", length(ns)))
    for (label, tn) in rows
        cells = [cell(label, tn, n) for n in ns]
        all(==("–"), cells) && continue
        println(io, "| `", label, "` ", tn, " | ", join(cells, " | "), " |")
    end
end

function summarize(io, rows, meta, name)
    println(io, "## ", name, "\n")
    for m in meta
        any(p -> startswith(m, p), ("commit", "cpu_model", "cpu_name", "threads", "grid", "Julia Version", "timestamp")) &&
            println(io, "- ", m)
    end
    D = Dict(key(r) => r.ns for r in rows)
    get_(l, lay, e, tn, d, n) = get(D, (l, lay, e, tn, d, n), NaN)

    for d in uniq([r.d for r in rows]), e in uniq([r.eltype for r in rows if r.d == d])
        sub = filter(r -> r.d == d && r.eltype == e, rows)
        any(r -> r.layout == "qudit", sub) && any(r -> r.layout == "generator", sub) || continue
        ns = uniq([r.n for r in sub])
        labs = labels_in(sub, r -> true)
        table(io, "### Qudit-major / generator-major, d = $d, $e", labs, ns,
              (l, tn, n) -> ratio(get_(l, "qudit", e, tn, d, n) / get_(l, "generator", e, tn, d, n)))
        # rows where the slower layout loses more than a switch round trip: then
        # switching for that operation would pay off
        table(io, "### Layout difference beyond one switch round trip, d = $d, $e " *
                  "(QM − GM − switch if QM is slower, GM − QM − switch if GM is slower; – when within a switch)",
              filter(x -> x[1] != "switch", labs), ns,
              (l, tn, n) -> begin
                  q = get_(l, "qudit", e, tn, d, n); g = get_(l, "generator", e, tn, d, n)
                  s = q > g ? get_("switch", "generator", e, tn, d, n) : get_("switch", "qudit", e, tn, d, n)
                  x = abs(q - g) - s
                  isnan(x) || x <= 0 ? "–" : (q > g ? "QM +" : "GM +") * fmt(x)
              end)
    end

    bits = filter(r -> r.d == 2 && r.eltype == "Bits", rows)
    if !isempty(bits)
        for lay in uniq([r.layout for r in bits])
            sub = filter(r -> r.layout == lay, bits)
            ns = uniq([r.n for r in sub])
            table(io, "### Bits / UInt8, d = 2, $lay-major", labels_in(sub, r -> true), ns,
                  (l, tn, n) -> ratio(get_(l, lay, "Bits", tn, 2, n) / get_(l, lay, "UInt8", tn, 2, n)))
        end
    end
end

function compare(io, a, b, na, nb)
    Db = Dict(key(r) => r.ns for r in b)
    println(io, "\n## ", nb, " / ", na, "\n")
    configs = unique([(r.layout, r.eltype, r.d) for r in a])
    for (lay, e, d) in configs
        sub = filter(r -> r.layout == lay && r.eltype == e && r.d == d, a)
        ns = uniq([r.n for r in sub])
        Da = Dict(key(r) => r.ns for r in sub)
        table(io, "### $lay-major, $e, d = $d", labels_in(sub, r -> true), ns,
              (l, tn, n) -> ratio(get(Db, (l, lay, e, tn, d, n), NaN) / get(Da, (l, lay, e, tn, d, n), NaN)))
    end
    rs = [Db[key(r)] / r.ns for r in a if haskey(Db, key(r))]
    isempty(rs) || @printf(io, "\nGeometric mean of %s / %s over %d rows: %.3f\n", nb, na, length(rs),
                           exp(sum(log, rs) / length(rs)))
end

function main(args)
    1 <= length(args) <= 2 || error("usage: julia summarize.jl a.csv [b.csv]")
    a, ma = load(args[1])
    summarize(stdout, a, ma, basename(args[1]))
    if length(args) == 2
        b, mb = load(args[2])
        println()
        summarize(stdout, b, mb, basename(args[2]))
        compare(stdout, a, b, basename(args[1]), basename(args[2]))
    end
end

main(ARGS)
