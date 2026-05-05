"""
biot_circle_gpu.jl

GPU demo: 2D circle in external flow using LilyBiotSim (LilyFlow + BiotSavartPoisson).
Compares one-step velocity field vs plain LilyPadSim, then times both to t=50.

Run with:
    julia --project=. examples/biot_circle_gpu.jl
"""

using WaterLily
using LilyPad
using BiotSavartBCs
using CUDA
using Plots
using Statistics

# ── Setup ────────────────────────────────────────────────────────────────────

CUDA.functional() || error("No CUDA GPU found — this demo requires a CUDA device")
mem = CUDA.CuArray
println("CUDA device: ", CUDA.name(CUDA.device()))
println("ν = 0.0 (inviscid)")

n, m = 3 * 2^6, 2^7   # 2× README circle resolution

function make_circle(n, m, ctor; mem=Array, T=Float32)
    radius, center = T(m / 8), T(m / 2 - 1)
    sdf(x, t) = sqrt(sum(abs2, x .- center)) - radius
    ctor((n, m), (1, 0), 2radius; ν=0, body=AutoBody(sdf), mem, T)
end

lb = make_circle(n, m, LilyBiotSim; mem)
lp = make_circle(n, m, LilyPadSim;  mem)
println("LilyBiotSim: $(typeof(lb.flow)) + $(typeof(lb.pois))")
println("LilyPadSim:  $(typeof(lp.flow)) + $(typeof(lp.pois))")

# ── One-step snapshot ────────────────────────────────────────────────────────

function vorticity_field(sim)
    a = sim.flow.σ
    @WaterLily.inside a[I] = WaterLily.curl(3, I, sim.flow.u)
    Array(@view a[WaterLily.inside(a)])
end

sim_step!(lb)
sim_step!(lp)
CUDA.synchronize()

u_lb = Array(lb.flow.u[:, :, 1])
u_lp = Array(lp.flow.u[:, :, 1])

out_dir = joinpath(@__DIR__, "..", "assets")
mkpath(out_dir)

clims = extrema([u_lb; u_lp])
v1 = contourf(u_lb', title="LilyBiotSim u_x (one step)", xlabel="x", ylabel="y", clims=clims)
v2 = contourf(u_lp', title="LilyPadSim u_x (one step)", xlabel="x", ylabel="y", clims=clims)
vel_fig = plot(v1, v2, layout=(1, 2), size=(1100, 420))
savefig(vel_fig, joinpath(out_dir, "biot_circle_one_step.png"))
println("One-step snapshot saved → assets/biot_circle_one_step.png")

# Smoke-test vorticity right after first step — fail fast before the long run
function vorticity_field(sim)
    a = sim.flow.σ
    @WaterLily.inside a[I] = WaterLily.curl(3, I, sim.flow.u)
    Array(@view a[WaterLily.inside(a)])
end
ω_test = vorticity_field(lb)
println("Vorticity smoke-test OK — size: ", size(ω_test), "  range: ",
        round(minimum(ω_test), digits=3), " to ", round(maximum(ω_test), digits=3))

# ── Timed run to t=50 ─────────────────────────────────────────────────────────

function timed_run!(sim; t_end=50.0, dt_out=0.05)
    ts    = Float64[]
    drag  = Float64[]
    lift  = Float64[]
    CUDA.synchronize()
    t_wall = time()
    for t in dt_out:dt_out:t_end
        sim_step!(sim, t; remeasure=false)
        f = WaterLily.pressure_force(sim) ./ (0.5 * sim.L * sim.U^2)
        push!(ts,   t)
        push!(drag, f[1])
        push!(lift, f[2])
    end
    CUDA.synchronize()
    elapsed = time() - t_wall
    ts, drag, lift, elapsed
end

println("\nRunning LilyBiotSim to t=50...")
ts_lb, drag_lb, lift_lb, t_lb = timed_run!(lb)

println("Running LilyPadSim  to t=50...")
ts_lp, drag_lp, lift_lp, t_lp = timed_run!(lp)

println("\n=== Timing (wall time to tU/L = 50) ===")
println("LilyBiotSim: $(round(t_lb, digits=2)) s")
println("LilyPadSim:  $(round(t_lp, digits=2)) s")
println("Biot/Plain ratio: $(round(t_lb/t_lp, digits=2))×")

# ── Force history plot ────────────────────────────────────────────────────────

pf = plot(ts_lb, drag_lb, label="drag  BiotLP", xlabel="tU/L", ylabel="Pressure force coeff")
plot!(pf, ts_lp, drag_lp, label="drag  LP",    ls=:dash)
plot!(pf, ts_lb, lift_lb, label="lift  BiotLP")
plot!(pf, ts_lp, lift_lp, label="lift  LP",    ls=:dash)
savefig(pf, joinpath(out_dir, "biot_circle_forces.png"))
println("Force history saved → assets/biot_circle_forces.png")

# ── Final vorticity plots ─────────────────────────────────────────────────────
# Use WaterLily.flood (contourf wrapper) + body_plot!, matching the WaterLily examples style.
# vorticity_field already wrote into flow.σ; flood reads it before body_plot! overwrites σ.

function vort_panel(sim, title_str)
    R = WaterLily.inside(sim.flow.p)
    # @inside with curl(3,...) doesn't compile for 2D GPU arrays; compute on CPU
    u_cpu = Array(sim.flow.u)
    σ_cpu = zeros(eltype(u_cpu), size(sim.flow.p))
    @WaterLily.inside σ_cpu[I] = WaterLily.curl(3, I, u_cpu) * sim.L / sim.U
    p = WaterLily.flood(σ_cpu[R]; clims=(-5,5), cfill=:seismic,
                        legend=false, title=title_str, axis=([], false), border=:none)
    # draw circle body outline manually
    radius = Float32(m / 8); center = Float32(m / 2 - 1) - 1  # -1 for inside offset
    θ = range(0, 2π; length=100)
    WaterLily.addbody(center .+ radius .* cos.(θ), center .+ radius .* sin.(θ))
    p
end

wv1 = vort_panel(lb, "LilyBiotSim ω_z (t=50)")
wv2 = vort_panel(lp, "LilyPadSim ω_z (t=50)")
vort_fig = plot(wv1, wv2, layout=(1, 2), size=(1100, 440))
savefig(vort_fig, joinpath(out_dir, "biot_circle_vorticity.png"))
println("Vorticity plot saved → assets/biot_circle_vorticity.png")

# ── Settled-regime metrics (t = 25–50) ───────────────────────────────────────

mask = ts_lb .>= 25.0
mask_lp = ts_lp .>= 25.0

println("\n=== Force metrics (t = 25–50) ===")
println("LilyBiotSim  drag mean = $(round(mean(drag_lb[mask]), digits=4))")
println("LilyPadSim   drag mean = $(round(mean(drag_lp[mask_lp]), digits=4))")
println("LilyBiotSim  lift mean = $(round(mean(lift_lb[mask]), digits=4))  (expected ≈ 0)")
println("LilyBiotSim  drag std  = $(round(std(drag_lb[mask]), digits=4))")
println("LilyPadSim   drag std  = $(round(std(drag_lp[mask_lp]), digits=4))")
