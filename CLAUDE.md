# LilyPad.jl

## Agent Routing Note
- This repo's local CLAUDE file is this project-root `CLAUDE.md`.
- Read the global `c:\Users\gweymouth\.claude\CLAUDE.md` first, then this local file.
- Write top-level persistent directives to the global `c:\Users\gweymouth\.claude\CLAUDE.md` file.
- On conversation compaction or resume, reread both the global file and this project-root `CLAUDE.md` before continuing.


## Package Purpose
This package extends the WaterLily.jl package with a semi-implicit time stepping scheme to increase simulation speed for interactive simulations. 

## Key Types

- `LilyPadSim <: WaterLily.AbstractSimulation` — thin wrapper around `WaterLily.Simulation` that forwards properties and dispatches `sim_step!` to `semi_mom_step!`
- `WaterLily.Flow` — reused unchanged; `flow.f` holds SL-advected velocity, `flow.σ` scratch for ∂ᵢp per component
- `WaterLily.AbstractPoisson` / `MultiLevelPoisson` — reused unchanged for pressure projection

## Entry Points

- `LilyPadSim(dims, uBC, L; fixed_dt=nothing, kwargs...)` — constructor; same kwargs as `WaterLily.Simulation` with optional legacy fixed-step control
- `sim_step!(sim::LilyPadSim[, t_end]; kwargs...)` — advance simulation

## Source Layout

- `src/LilyPad.jl` — module, `LilyPadSim` wrapper + `sim_step!(::LilyPadSim)` override, includes, exports
- `src/SemiAdvect.jl` — `departure`, `scalar_advect!`
- `src/SemiMomStep.jl` — `semi_mom_step!` + helpers
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
not yet been validated against the legacy reference.

## Scheme Notes (Semi-Lagrangian two-step)
- `flow.σ` reused per component in corrector (overwritten each `i`); no extra large arrays
- Diffusion omitted in v0.1 (inviscid semi-Lagrangian)
- No CFL constraint from advection; `CFL(flow)` still computed and pushed for reference
- `departure` clamps the initial face location and both RK2 substeps to keep interpolation bounded for large `dt`

## Timestep Modes
- **Fixed mode** (`fixed_dt=1.5`): each `sim_step!` uses constant `dt`, matching legacy behavior
- **Adaptive mode** (`fixed_dt=nothing`): CFL-controlled stepping via `push!(flow.Δt, CFL(flow))`

## Legacy Timestep Reference
- Legacy repo: `../lily-pad/LilyPad/BDIM.pde`
- `dt == 0` means adaptive mode via `setDt()` and `checkCFL()`
- Main legacy circle demo: `../lily-pad/LilyPad/LilyPad.pde`
- The legacy circle demo constructs `BDIM(n,n,1.5,body)`, so the reference semi-Lagrangian circle run uses a constant `dt = 1.5`
- Legacy adaptive CFL helper is `min(u.CFL(nu), 1)`, but that is not the default path used by the main semi-Lagrangian circle example

## Testing Status
**Passing tests do NOT validate algorithm correctness.** Tests exist for:
- SemiAdvect: departure RK2 structure, scalar component advection on staggered faces, boundary-adjacent transport checks
- SemiMomStep: one-step uniform-flow fixed-point behavior, divergence finiteness after projection, fixed-step append behavior
- LilyPadSim: one-step wrapper parity with WaterLily `Simulation`, timestep progression, fixed-dt consistency

**Unvalidated against reference:**
- Full force-history parity with WaterLily at `dt=1.5`
- Energy behavior and conservation properties over long horizons

## Practical Notes on Timestep Stability
- Departure clamping keeps interpolation finite through `dt=1.6`
- Numerical quality clearly degrades at large `dt`, but tests don't capture this
- `dt=0.8` shows acceptable behavior in spot checks; `dt=1.5` used for legacy compatibility

## Critical Algorithm Contracts
- **BDIM absolute-input form (current implementation)**: `BDIM!` consumes absolute semi-Lagrangian advected velocity in `flow.f` (not increment form). The helper subtracts `flow.V` internally, then applies `μddn + V + μ₀*f`.
- **Corrector pressure term**: Pressure gradient in corrector must be advected and subtracted: `F.advect(...) - dp.advect(...)`

## Current Status & Blockers
- **Interpolation asymmetry fix landed**: semi-Lagrangian component interpolation now uses nearest-centered quadratic (`quad`) to avoid mirror-bias from floor-centered stencils.
- **Advection interpolation reduction landed**: corrector now reuses the averaged velocity in `flow.u`, reducing total vector-interp calls from 6 to 4 per advected point (predictor 2 + corrector 2 instead of predictor 2 + corrector 4).
- **README path simplified**: `examples/readme_circle_compare.jl` is the canonical example and emits one-step velocity + long-run force trace figures.
- **Current force status (inviscid comparison, LP `fixed_dt=1.5`, metrics over t=25–50)**:
	- lift mean absolute error (`mean(WL) - mean(LP)`): `-0.1031` (expected mean lift is near zero; do not use relative error)
	- drag mean relative error (`1 - mean(LP)/mean(WL)`): `0.2472` (~25% lower mean drag in LP)
	- drag std  relative error (`1 - std(LP)/std(WL)`):  `-0.0378` (~3.8% larger LP amplitude)
	- lift std  relative error (`1 - std(LP)/std(WL)`):  `-0.0170` (~1.7% larger LP amplitude)
- **3D GPU profiling status (inviscid, synchronized timing)**:
	- `fixed_dt=1.5`: `sec/(tU/L) = 0.323629`; advection share `0.610`; projection share `0.324`.
	- `fixed_dt=2.5`: `sec/(tU/L) = 0.181698`; advection share `0.586`; projection share `0.348`.
	- Divergence and projection iterations stayed essentially unchanged with increase dt.
- **Reconstruction tradeoff status**: replacing face-aware quadratic reconstruction with face-aware linear `interp` did **not** produce a meaningful GPU speedup after repeated synchronized timings (LP `sec/(tU/L)`: `0.303479` interp vs `0.309052` quad at `dt=1.5`; `0.203489` interp vs `0.201559` quad at `dt=2.5`). The speed difference is noise-level, while 2D README-circle force amplitudes degrade badly (`drag std` error ~`0.496`, `lift std` error ~`0.160`). Conclusion: swapping `quad` for `interp` is not a worthwhile optimization.
- **Allocation note**: the README circle case still shows nontrivial allocations during body remeasurement in both WL and LP. This is likely a separate type-stability or body-measurement issue and is deferred for later investigation.
- **Current allocation conclusion**: on the README circle case with `remeasure=false`, LP allocates more per step than WL because its larger fixed timestep enters `project!` with larger pre-projection divergence/residual and therefore requires more multigrid iterations. This allocation gap is in the WaterLily Poisson solve path, not in LP's advection kernels.
- **Current allocation-testing rule**: for the README circle case, measure after initialization with `remeasure=false` and compare allocations per unit simulated time (`tU/L`), not per step. On that metric LP is currently better because it advances farther per step.
- **Profiling note**: recent evidence shows interpolation-heavy advection kernels are the primary GPU bottleneck at tested timesteps; projection remains important but was previously over-attributed due to timing methodology. The corrector interp-count reduction was worthwhile, but further reconstruction simplification appears to be in diminishing-returns territory.
- **README dt-sweep status**: LP `fixed_dt` in the `1.5-2.5` range currently looks like the best speed/accuracy tradeoff on the README circle case; `dt` above ~`2.5` can look faster but shows unstable/error-sign-flipping behavior.
- **Interpretation**: oscillation amplitudes match well (~1-2%); mean drag offset (~25%) is the primary open correctness question.
- **Near-term execution order**: allocation checks first, then CPU README benchmark/optimization, then 3D GPU bring-up.
- **Constraint**: treat performance numbers as provisional until each stage's correctness checks pass.

## Active Work
The primary development goal is to port the key capabilities of the legacy lily-pad code over to the new WaterLily.jl ecosystem.

**Tomorrow Plan (Explicit):**
1. **SIMD allocation parity tests (LP vs WL)**
	- Create focused allocation tests in SIMD mode for the same stepping scenarios in LP and WL.
	- Record allocations after initialization with `remeasure=false`.
	- For the README circle case, compare allocations per unit simulated time (`tU/L`) rather than per step.
	- Current finding: LP is worse per step but better per simulated time because it uses a larger timestep.

2. **README circle benchmark + CPU optimization (LP vs WL)**
	- Use `examples/readme_circle_compare.jl` setup as the benchmark scenario.
	- Benchmark both solvers with identical runtime settings and report both timing and force metrics.
	- Include timestep-tradeoff discussion: larger LP `dt` reduces step count but increases pre-projection divergence and Poisson iterations.
	- Correctness gate before optimization claims: preserve current force-behavior expectations (drag/lift summary metrics over `t=25-50`) and no regressions in `test/runtests.jl`.
	- Only after passing the correctness gate, optimize hotspots and re-benchmark.

3. **3D GPU sphere case: run, validate, then optimize**
	- Build a 3D sphere case on GPU (`CuArray`) with a WL baseline and an LP run under matched physical settings.
	- First goal: successful compile and execution.
	- Second goal: correctness checks (same levels as the 2D CPU tests above).
	- Third goal: optimize speed only after correctness checks pass.

**Do-not-skip gates (tomorrow):**
- No performance claim without corresponding correctness evidence.
- No optimization pass before the stage-specific correctness checks pass.
- Keep README/CLAUDE metrics and statements synchronized with latest validated runs.

**Immediate next goal (today):**
- Benchmark and profile LP vs WL on the README circle case, using per-simulated-time cost as the primary efficiency metric and keeping the timestep/Poisson-iteration tradeoff explicit.
