using CUDA
using WaterLily
using LilyPad
using Printf
using Statistics

import WaterLily: BC!, project!, CFL, exitBC!, @loop, inside, @inside, div
import LilyPad: predict_advect!, correct_advect!

mutable struct StepProfile
    advection::Float64
    projection::Float64
    other::Float64
end

StepProfile() = StepProfile(0.0, 0.0, 0.0)

function Base.:+(a::StepProfile, b::StepProfile)
    StepProfile(a.advection + b.advection, a.projection + b.projection, a.other + b.other)
end

total_time(p::StepProfile) = p.advection + p.projection + p.other

mutable struct StepDiagnostics
    predictor_div_rms::Float64
    corrector_div_rms::Float64
    predictor_iters::Vector{Int}
    corrector_iters::Vector{Int}
end

StepDiagnostics() = StepDiagnostics(NaN, NaN, Int[], Int[])

function sphere_lp(n, m; ν=0f0, U=1, T=Float32, mem=CuArray, fixed_dt=1.5f0)
    radius, center = m / 8, m / 2 - 1
    body = AutoBody((x, t) -> sqrt(sum(abs2, x .- center)) - radius)
    LilyPadSim((n, m, m), (U, 0, 0), 2radius; ν, body, T, mem, fixed_dt)
end

function timed_sync(f)
    CUDA.synchronize()
    return @elapsed begin
        f()
        CUDA.synchronize()
    end
end

function interior_view(a)
    ranges = ntuple(d -> 2:(size(a, d) - 1), ndims(a))
    return @view a[ranges...]
end

function divergence_rms!(flow, pois)
    @inside pois.z[I] = div(I, flow.u)
    CUDA.synchronize()
    z = Array(interior_view(pois.z))
    return sqrt(sum(abs2, z) / length(z))
end

function project_with_iters!(flow, pois, weight)
    if pois.n isa AbstractVector
        len0 = length(pois.n)
        elapsed = timed_sync(() -> project!(flow, pois, weight))
        vals = Int.(Array(pois.n))[(len0 + 1):end]
        return elapsed, isempty(vals) ? 0 : sum(vals)
    else
        elapsed = timed_sync(() -> project!(flow, pois, weight))
        return elapsed, Int(pois.n[])
    end
end

function lp_profiled_step!(sim::LilyPadSim, diag::StepDiagnostics)
    flow, pois = sim.flow, sim.pois
    flow.u⁰ .= flow.u
    dt = isnothing(sim.fixed_dt) ? flow.Δt[end] : convert(eltype(flow.p), sim.fixed_dt)
    update_cfl = isnothing(sim.fixed_dt)

    other = timed_sync(() -> (flow.Δt[end] = dt))
    t₁ = sum(flow.Δt)

    advection = timed_sync(() -> predict_advect!(flow, dt))
    other += timed_sync(() -> begin
        LilyPad.BDIM!(flow)
        BC!(flow.u, flow.uBC, flow.exitBC, flow.perdir, t₁)
        flow.exitBC && exitBC!(flow.u, flow.u⁰, dt)
    end)

    isnan(diag.predictor_div_rms) && (diag.predictor_div_rms = divergence_rms!(flow, pois))
    pred_projection, pred_iters = project_with_iters!(flow, pois, 1)
    projection = pred_projection
    push!(diag.predictor_iters, pred_iters)

    other += timed_sync(() -> BC!(flow.u, flow.uBC, flow.exitBC, flow.perdir, t₁))

    advection += timed_sync(() -> correct_advect!(flow, dt))
    other += timed_sync(() -> begin
        LilyPad.BDIM!(flow)
        BC!(flow.u, flow.uBC, flow.exitBC, flow.perdir, t₁)
    end)

    isnan(diag.corrector_div_rms) && (diag.corrector_div_rms = divergence_rms!(flow, pois))
    corr_projection, corr_iters = project_with_iters!(flow, pois, 0.5)
    projection += corr_projection
    push!(diag.corrector_iters, corr_iters)

    other += timed_sync(() -> begin
        BC!(flow.u, flow.uBC, flow.exitBC, flow.perdir, t₁)
        push!(flow.Δt, update_cfl ? CFL(flow) : dt)
    end)

    return StepProfile(advection, projection, other)
end

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

function print_stats(label, values)
    println(label, " min/mean/max = ",
            (minimum(values), mean(values), maximum(values)))
end

function summarize_dt(fixed_dt; n=3 * 2^5, m=2^6, t_end=50.0, ν=0f0)
    sim = sphere_lp(n, m; fixed_dt=fixed_dt, ν=ν)
    sim_step!(sim)

    elapsed, advanced, steps, profile, diag = run_profiled_window!(sim; t_end=t_end)
    total = total_time(profile)

    println("--- LP dt = ", fixed_dt, " ---")
    println("steps = ", steps)
    println(@sprintf("advanced tU/L = %.6f", advanced))
    println(@sprintf("sec/(tU/L) = %.6f", elapsed / advanced))
    println(@sprintf("advection sec/share = %.6f / %.3f", profile.advection, profile.advection / total))
    println(@sprintf("projection sec/share = %.6f / %.3f", profile.projection, profile.projection / total))
    println(@sprintf("other sec/share = %.6f / %.3f", profile.other, profile.other / total))
    println(@sprintf("initial predictor div RMS = %.6e", diag.predictor_div_rms))
    println(@sprintf("initial corrector div RMS = %.6e", diag.corrector_div_rms))
    print_stats("predictor projection cycles", diag.predictor_iters)
    print_stats("corrector projection cycles", diag.corrector_iters)
    println()
end

function main()
    @assert CUDA.functional()
    println("Threads.nthreads() = ", Threads.nthreads())
    println("WaterLily backend = ", WaterLily.backend)
    println("Viscosity ν = 0.0")
    println()

    summarize_dt(1.5f0)
    summarize_dt(2.5f0)
end

main()