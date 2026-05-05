using CUDA
using WaterLily
using LilyPad
using Printf
using Statistics

import WaterLily: scale_u!

# ---------------------------------------------------------------------------
# Profiling helpers
# ---------------------------------------------------------------------------

mutable struct StepProfile
    advection  :: Float64
    projection :: Float64
    other      :: Float64
end

StepProfile() = StepProfile(0.0, 0.0, 0.0)

function Base.:+(a::StepProfile, b::StepProfile)
    StepProfile(a.advection + b.advection, a.projection + b.projection, a.other + b.other)
end

total_time(p::StepProfile) = p.advection + p.projection + p.other

mutable struct StepDiagnostics
    predictor_div_rms :: Float64
    corrector_div_rms :: Float64
    predictor_iters   :: Vector{Int}
    corrector_iters   :: Vector{Int}
end

StepDiagnostics() = StepDiagnostics(NaN, NaN, Int[], Int[])

# ---------------------------------------------------------------------------
# Simulation constructor
# ---------------------------------------------------------------------------

function sphere_lp(n, m; ν=0f0, U=1, T=Float32, mem=CuArray, Δt=1.5f0)
    radius, center = m / 8, m / 2 - 1
    body = AutoBody((x, t) -> sqrt(sum(abs2, x .- center)) - radius)
    LilyPadSim((n, m, m), (U, 0, 0), 2radius; ν, body, T, mem, Δt)
end

# ---------------------------------------------------------------------------
# Low-level utilities
# ---------------------------------------------------------------------------

function timed_sync(f)
    CUDA.synchronize()
    return @elapsed begin
        f()
        CUDA.synchronize()
    end
end

function divergence_rms!(flow, pois)
    @inside pois.z[I] = WaterLily.div(I, flow.u)
    CUDA.synchronize()
    interior = ntuple(d -> 2:(size(pois.z, d) - 1), ndims(pois.z))
    z = Array(@view pois.z[interior...])
    return sqrt(sum(abs2, z) / length(z))
end

# Returns (elapsed, iteration_count) for one mom_project! call
function project_timed!(flow, pois, w, t)
    n0 = pois.n isa AbstractVector ? length(pois.n) : nothing
    elapsed = timed_sync(() -> WaterLily.mom_project!(flow, pois, w, t))
    iters = if pois.n isa AbstractVector
        vals = Int.(Array(pois.n))[(n0 + 1):end]
        isempty(vals) ? 0 : sum(vals)
    else
        Int(pois.n[])
    end
    return elapsed, iters
end

# ---------------------------------------------------------------------------
# Profiled step: same sequence as mom_step!, with per-phase timing
# ---------------------------------------------------------------------------

function lp_profiled_step!(sim::WaterLily.AbstractSimulation, diag::StepDiagnostics)
    flow, pois = sim.flow, sim.pois

    # Replicate mom_step! preamble: save u⁰ and zero interior u
    other = timed_sync(() -> begin
        flow.u⁰ .= flow.u
        scale_u!(flow, 0)
    end)
    t₁ = sum(flow.Δt)
    t₀ = t₁ - flow.Δt[end]

    # Predictor advection
    advection = timed_sync(() -> WaterLily.mom_predict!(flow, t₀, t₁))

    # Predictor projection
    isnan(diag.predictor_div_rms) && (diag.predictor_div_rms = divergence_rms!(flow, pois))
    pred_proj, pred_iters = project_timed!(flow, pois, 1, t₁)
    projection = pred_proj
    push!(diag.predictor_iters, pred_iters)

    # Corrector advection
    advection += timed_sync(() -> WaterLily.mom_correct!(flow, t₁))

    # Corrector projection
    isnan(diag.corrector_div_rms) && (diag.corrector_div_rms = divergence_rms!(flow, pois))
    corr_proj, corr_iters = project_timed!(flow, pois, 0.5, t₁)
    projection += corr_proj
    push!(diag.corrector_iters, corr_iters)

    # Timestep push (CFL override returns Δt[end] unchanged for LilyFlow)
    other += timed_sync(() -> push!(flow.Δt, WaterLily.CFL(flow)))

    return StepProfile(advection, projection, other)
end

# ---------------------------------------------------------------------------
# Profiling window
# ---------------------------------------------------------------------------

function run_profiled_window!(sim; t_end=50.0)
    profile = StepProfile()
    diag = StepDiagnostics()
    t0 = sim_time(sim)
    s0 = length(sim.flow.Δt)

    CUDA.synchronize()
    elapsed = @elapsed begin
        while sim_time(sim) < t_end
            profile = profile + lp_profiled_step!(sim, diag)
        end
        CUDA.synchronize()
    end

    advanced = sim_time(sim) - t0
    steps = length(sim.flow.Δt) - s0
    return elapsed, advanced, steps, profile, diag
end

# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

function print_stats(label, values)
    println(label, " min/mean/max = ",
            (minimum(values), mean(values), maximum(values)))
end

function summarize_dt(Δt; n=3 * 2^5, m=2^6, t_end=50.0, ν=0f0)
    sim = sphere_lp(n, m; Δt=Δt, ν=ν)
    sim_step!(sim)   # warm-up / initial transient

    elapsed, advanced, steps, profile, diag = run_profiled_window!(sim; t_end=t_end)
    total = total_time(profile)

    println("--- LP Δt = ", Δt, " ---")
    println("steps = ", steps)
    println(@sprintf("advanced tU/L = %.6f", advanced))
    println(@sprintf("sec/(tU/L) = %.6f", elapsed / advanced))
    println(@sprintf("advection  sec/share = %.6f / %.3f", profile.advection,  profile.advection  / total))
    println(@sprintf("projection sec/share = %.6f / %.3f", profile.projection, profile.projection / total))
    println(@sprintf("other      sec/share = %.6f / %.3f", profile.other,      profile.other      / total))
    println(@sprintf("initial predictor div RMS = %.6e", diag.predictor_div_rms))
    println(@sprintf("initial corrector div RMS = %.6e", diag.corrector_div_rms))
    print_stats("predictor projection cycles", diag.predictor_iters)
    print_stats("corrector projection cycles", diag.corrector_iters)
    println()
end

function main()
    @assert CUDA.functional()
    println("Threads.nthreads() = ", Threads.nthreads())
    println("WaterLily backend  = ", WaterLily.backend)
    println("Viscosity ν = 0.0 (inviscid)")
    println()

    summarize_dt(1.5f0)
    summarize_dt(2.5f0)
end

main()
