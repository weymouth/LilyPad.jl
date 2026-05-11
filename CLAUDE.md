# LilyPad.jl

## Agent Routing Note
- This repo's local CLAUDE file is this project-root `CLAUDE.md`.
- Read the global `c:\Users\gweymouth\.claude\CLAUDE.md` first, then this local file.
- Write top-level persistent directives to the global `c:\Users\gweymouth\.claude\CLAUDE.md` file.
- On conversation compaction or resume, reread both the global file and this project-root `CLAUDE.md` before continuing.


## Package Purpose
This package extends the WaterLily.jl package with a semi-implicit time stepping scheme to increase simulation speed for interactive simulations. 

## Key Types

- `LilyFlow{D,T,...} <: WaterLily.AbstractFlow{D,T}` — thin wrapper around `WaterLily.Flow`; overrides `mom_predict!`, `mom_correct!`, `CFL`, and `BDIM!` with semi-Lagrangian logic; all field access forwards to the inner `flow` via `getproperty`/`setproperty!`
- `LilyPadSim` — plain function (constructor alias), not a struct; returns `WaterLily.Simulation` built with `flow_ctor=LilyFlow`
- `WaterLily.AbstractPoisson` / `MultiLevelPoisson` — reused unchanged for pressure projection

## Entry Points

- `LilyPadSim(dims, uBC, L; Δt=1.5, kwargs...)` — same kwargs as `WaterLily.Simulation`; `Δt` sets the fixed timestep
- `sim_step!(sim[, t_end]; kwargs...)` — advance simulation (dispatches through `WaterLily.Simulation`)

## Source Layout

- `src/LilyPad.jl` — module entry point; `LilyPadSim` constructor alias, includes, exports
- `src/LilyFlow.jl` — `LilyFlow` struct + `getproperty`/`setproperty!` forwarding + `mom_predict!` / `mom_correct!` / `CFL` / `BDIM!` overrides
- `src/SemiAdvect.jl` — `departure` (RK2) and `scalar_advect!`
- `src/util.jl` — `quad` interpolation helpers
- `test/runtests.jl` — backend dispatch (mirrors WaterLily)
- `test/maintests.jl` — @testsets for each source file

## Environment Notes
A legacy code is in the ../lily-pad/ folder. The WaterLily source is in ../WaterLily.jl/.
WaterLily version: 1.6.1 (uuid: ed894a53-35f9-47f1-b17f-85db9237eebd)

## Critical Comparison Guardrail (Do Not Skip)
- **Always match viscosity between WL and LP before any timing or solver comparison.**
- **For inviscid comparisons, set `ν=0` explicitly for both solvers. Do not rely on defaults.**
- If using `Re`-based setup, confirm both cases map to the same `ν` numerically before running.
- Every benchmark/profiling script should print `ν` in its header and report it with results.
- Any result produced with mismatched `ν` is invalid and must be rerun.

## Development Order (Required)
Always execute work in this order:
1. Algorithm validation first
2. Implementation decisions second
3. Correctness testing third
4. Benchmarking last

Benchmarking is blocked until the first three stages are explicitly checked off.
Do not use performance results to justify an algorithm or method variant that has
not yet been considered and discussed.

## Testing Status
**Passing tests do NOT validate algorithm correctness.** Tests exist for:
- SemiAdvect: departure RK2 structure, scalar component advection on staggered faces, boundary-adjacent transport checks
- LilyFlow: one-step uniform-flow fixed-point behavior, divergence finiteness after projection, second-step regression (guards against scale_u! bug)
- LilyPadSim: one-step wrapper parity with WaterLily `Simulation`, timestep progression, second-step regression

Test suite runs on `Array` and `CuArray` backends. CUDA is an optional test dependency; GPU tests are skipped if `CUDA.functional()` returns false.

## Scheme Notes (Semi-Lagrangian two-step)
- Diffusion omitted in v0.1 (inviscid semi-Lagrangian)
- No CFL constraint from advection; `CFL(flow::LilyFlow)` returns `Δt[end]` unchanged to keep the step fixed
- `Δt` set at construction time is the user knob for speed/accuracy tradeoff; larger `Δt` is faster but less accurate

## Timestep Control
- `Δt` is passed through `LilyPadSim(...; Δt=...)` → `WaterLily.Simulation` → `LilyFlow` → `Flow`
- `CFL(a::LilyFlow)` returns `a.Δt[end]` unchanged so `mom_step!`'s single `push!(Δt, CFL(a))` keeps the step fixed
- There is no adaptive mode — the semi-Lagrangian scheme always runs at the user-supplied `Δt`

## Legacy Timestep Reference
- Legacy repo: `../lily-pad/LilyPad/BDIM.pde`
- `dt == 0` means adaptive mode via `setDt()` and `checkCFL()`
- Main legacy circle demo: `../lily-pad/LilyPad/LilyPad.pde`
- The legacy circle demo constructs `BDIM(n,n,1.5,body)`, so the reference semi-Lagrangian circle run uses a constant `dt = 1.5`

## Current Status & Blockers
- **Interpolation asymmetry fix landed**: semi-Lagrangian component interpolation now uses nearest-centered quadratic (`quad`) to avoid mirror-bias from floor-centered stencils.
- **Advection interpolation reduction landed**: corrector now reuses the averaged velocity in `flow.u`, reducing total vector-interp calls from 6 to 4 per advected point (predictor 2 + corrector 2 instead of predictor 2 + corrector 4).
- **README path simplified**: `examples/readme_circle_compare.jl` is the canonical example and emits one-step velocity + long-run force trace figures.
- **Current force status (inviscid comparison, LP `Δt=1.5`, metrics over t=25–50)**:
	- lift mean absolute error (`mean(WL) - mean(LP)`): `-0.093` (expected mean lift is near zero; do not use relative error)
	- drag mean relative error (`1 - mean(LP)/mean(WL)`): `0.244` (~25% lower mean drag in LP)
	- drag std  relative error (`1 - std(LP)/std(WL)`):  `-0.048` (larger LP amplitude)
	- lift std  relative error (`1 - std(LP)/std(WL)`):  `-0.036` (larger LP amplitude)
- **3D GPU profiling status (inviscid, synchronized timing, post-refactor)**:
	- `Δt=1.5`: `sec/(tU/L) = 0.325953`; advection share `0.602`; projection share `0.387`.
	- `Δt=2.5`: `sec/(tU/L) = 0.278146`; advection share `0.399`; projection share `0.594`.
	- Divergence and projection iterations stayed essentially unchanged with increased dt.
- **Reconstruction tradeoff status**: replacing face-aware quadratic reconstruction with face-aware linear `interp` did **not** produce a meaningful GPU speedup after repeated synchronized timings (LP `sec/(tU/L)`: `0.303479` interp vs `0.309052` quad at `dt=1.5`; `0.203489` interp vs `0.201559` quad at `dt=2.5`). The speed difference is noise-level, while 2D README-circle force amplitudes degrade badly (`drag std` error ~`0.496`, `lift std` error ~`0.160`). Conclusion: swapping `quad` for `interp` is not a worthwhile optimization.
- **Allocation note**: the README circle case still shows nontrivial allocations during body remeasurement in both WL and LP. This is likely a separate type-stability or body-measurement issue and is deferred for later investigation.
- **Current allocation conclusion**: on the README circle case with `remeasure=false`, LP allocates more per step than WL because its larger fixed timestep enters `project!` with larger pre-projection divergence/residual and therefore requires more multigrid iterations. This allocation gap is in the WaterLily Poisson solve path, not in LP's advection kernels.
- **Current allocation-testing rule**: for the README circle case, measure after initialization with `remeasure=false` and compare allocations per unit simulated time (`tU/L`), not per step. On that metric LP is currently better because it advances farther per step.
- **Profiling note**: recent evidence shows interpolation-heavy advection kernels are the primary GPU bottleneck at tested timesteps; projection remains important but was previously over-attributed due to timing methodology. The corrector interp-count reduction was worthwhile, but further reconstruction simplification appears to be in diminishing-returns territory.
- **README dt-sweep status**: LP `Δt` in the `1.5-2.5` range currently looks like the best speed/accuracy tradeoff on the README circle case; `dt` above ~`2.5` can look faster but shows unstable/error-sign-flipping behavior.
- **Interpretation**: oscillation amplitudes match well (~1-2%); mean drag offset (~25%) is the primary open correctness question.
- **Near-term execution order**: allocation checks first, then CPU README benchmark/optimization, then 3D GPU bring-up.
- **Constraint**: treat performance numbers as provisional until each stage's correctness checks pass.

## Active Work
1. ✅ Refactored to WL `Flow-refactor` branch: `LilyFlow <: AbstractFlow{D,T}` thin wrapper, `LilyPadSim` constructor alias, `SemiMomStep.jl` deleted, `fixed_dt` removed.
2. ✅ GPU tests enabled: CUDA added as optional test dependency; closures in tests use `f0` literals to be GPU-compatible; `Pkg.test()` runs on both `Array` and `CuArray`.
3. ✅ Ensure working interoperability with BiotSavartBCs.jl capability for "unbounded" simulations.
4. Add viscous diffusion term with primary consideration on the time-step stability vs simulation speed for intermediate and high Re cases (>1k). Update WL vs LP benchmarks.
5. ✅ Develop real-time interactive 2D fluid simulation and vizualization example using ParticleViz and PixelBodies.