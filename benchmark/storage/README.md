# Storage benchmark

`run.jl` times every public tableau operation in each storage configuration,
and `summarize.jl` turns the CSV files into the tables that decide the storage
defaults. It needs only QuditClifford and the Julia standard library, and it is
separate from the AirspeedVelocity regression suite in `benchmark/`.

## 1. What the runs decide

QuditClifford stores the X and Z blocks of a tableau in one of two layouts, and
at `d = 2` in one of two element types:

- **Layout.** `layout = :generator` (generator-major, the default: the entries of
  one generator are contiguous) or `layout = :qudit` (qudit-major: the entries
  of one qudit across all generators are contiguous).
- **Element type at `d = 2`.** `UInt8` (the default, one byte per entry) or
  `eltype = Bits` (packed, 64 entries per `UInt64` word).

Which choice is faster depends on the operation, on `n` and `d`, and on the
memory system of the machine. The runs decide:

1. the default layout for each `d` (2, 3, 5 and 31 stand for qubits, small
   primes, and the larger moduli);
2. the default element type at `d = 2`.

The decision needs runs on several machines, in particular on the x86 Linux
nodes that large simulations run on.

## 2. Installing Julia without root (x86 Linux)

[juliaup](https://github.com/JuliaLang/juliaup) installs Julia into your home
directory:

```sh
curl -fsSL https://install.julialang.org | sh -s -- --yes
source ~/.bashrc          # or open a new shell; juliaup adds itself to PATH
juliaup add release       # the current release
juliaup add lts           # optional: the oldest supported version (1.10)
julia --version
```

If the login nodes have no internet access, install on a machine that has, and
copy `~/.juliaup` and `~/.julia/juliaup` across; or download a tarball from
<https://julialang.org/downloads/> and unpack it anywhere.

Julia keeps packages in `~/.julia`. If the home quota is small, point
`JULIA_DEPOT_PATH` to a larger file system before the first run:

```sh
export JULIA_DEPOT_PATH=/path/with/space/julia-depot
```

## 3. Getting the code

```sh
git clone https://github.com/maltepuetz/QuditClifford.jl.git
cd QuditClifford.jl
git checkout <commit>     # the commit you were asked to benchmark
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
```

Run the instantiate step on a node with internet access; the benchmark itself
needs none.

## 4. A single-core run

Timings are single-threaded. Give the run a whole node, or at least a whole
core with nothing else on it, and pin it to one core:

```sh
export JULIA_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
taskset -c 2 julia --project=. benchmark/storage/run.jl storage_$(hostname)_default.csv
```

Leave out `taskset` where it is not available. A short profile checks the setup
first (under 10 minutes):

```sh
QC_BENCH_PROFILE=quick taskset -c 2 julia --project=. benchmark/storage/run.jl storage_$(hostname)_quick.csv
```

The default profile takes about 1 hour 40 minutes on one core of a recent
laptop; allow up to twice that on an older server core. The grid can be narrowed with
environment variables (comma lists):

| variable | default | meaning |
|---|---|---|
| `QC_BENCH_PROFILE` | `default` | `default` or `quick`; sets the default sizes and budget |
| `QC_BENCH_LAYOUTS` | `generator,qudit` | layouts |
| `QC_BENCH_ELTYPES` | `default,Bits` | element types; `Bits` runs at `d = 2` only |
| `QC_BENCH_DS` | `2,3,5,31` | qudit dimensions |
| `QC_BENCH_NS` | `512,1024,2048,4096` (`quick`: `256,1024`) | numbers of qudits |
| `QC_BENCH_TYPES` | `Stab,Destab` | `StabilizerTableau`, `DestabilizerTableau` |
| `QC_BENCH_OPS` | all rows | row labels, as in the CSV |
| `QC_BENCH_BUDGET` | `1.0` (`quick`: `0.15`) | seconds per row; sets the repetitions |

Each CSV line is the median time of one operation (`median_ns`) over `reps`
repetitions on fresh states. The comment lines at the top record the CPU, the
Julia version, the thread count, the commit and the time.

To look at the results:

```sh
julia benchmark/storage/summarize.jl storage_host_default.csv
julia benchmark/storage/summarize.jl laptop.csv server.csv    # second / first, per row
```

## 5. Batch scheduler template

A template for a batch scheduler. Adapt the directive syntax to the scheduler
of your site (the lines below use a common `#SBATCH`-style syntax); what matters
is one task on an exclusive node, the environment variables, and enough wall
time.

```sh
#!/bin/bash
#SBATCH --job-name=qc-storage
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --time=05:00:00          # default profile; 00:30:00 for quick
#SBATCH --output=qc-storage-%j.log

export JULIA_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
# export JULIA_DEPOT_PATH=/path/with/space/julia-depot
# export QC_BENCH_PROFILE=quick

cd /path/to/QuditClifford.jl
CORE=2
if command -v taskset >/dev/null; then PIN="taskset -c $CORE"; else PIN=""; fi
$PIN julia --project=. benchmark/storage/run.jl "storage_$(hostname)_${QC_BENCH_PROFILE:-default}_$(date +%Y%m%d).csv"
```

Run the quick profile once first to check the setup, then the default profile.
If the node type varies between jobs, a second default run on another node of
the same type shows how much the numbers move.

## 6. Sending the results back

Send the CSV files (one per run) and, if possible, the job log. The CSV header
records the CPU model and the commit, so nothing else is needed; please add the
node type in a line of text if the CPU model string does not identify it. The
files are small and plain text: attach them to an issue or an e-mail, or paste
them into a gist.
