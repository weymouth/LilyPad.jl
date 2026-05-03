using WaterLily
using LilyPad
using Plots
using Statistics
    
function circle_wl(n, m; U=1)
    radius, center = m / 8, m / 2 - 1
    sdf(x, t) = sqrt(sum(abs2, x .- center)) - radius
    Simulation((n, m), (U, 0), 2radius; ν=0, body=AutoBody(sdf))
end

function circle_lp(n, m; U=1)
    radius, center = m / 8, m / 2 - 1
    sdf(x, t) = sqrt(sum(abs2, x .- center)) - radius
    LilyPadSim((n, m), (U, 0), 2radius;
               ν=0,
               body=AutoBody(sdf),
               fixed_dt=1.5)
end

n, m = 3 * 2^5, 2^6
wl = circle_wl(n, m)
lp = circle_lp(n, m)
out_dir = joinpath(@__DIR__, "..", "assets")
mkpath(out_dir)

println("WL startup dt = ", wl.flow.Δt[end])
println("LP fixed_dt = 1.5")

# One full step for field snapshots
sim_step!(wl)
sim_step!(lp)

u_wl = Array(wl.flow.u[:, :, 1])
u_lp = Array(lp.flow.u[:, :, 1])

# 1) velocity x-component
v1 = contourf(u_wl', title="WaterLily u_x after one step", xlabel="x", ylabel="y")
v2 = contourf(u_lp', title="LilyPad u_x after one step", xlabel="x", ylabel="y")
vel_fig = plot(v1, v2, layout=(1, 2), size=(1100, 420))
savefig(vel_fig, joinpath(out_dir, "circle_one_step_compare.png"))

println("=== One-step WL vs LP velocity snapshot saved ===")

# 4) long-run force history comparison
function get_force_history!(sim; t_end=50.0, dt_out=0.1)
    ts = collect(1.0:dt_out:t_end)
    drag = Float64[]
    lift = Float64[]
    for t in ts
        sim_step!(sim, t; remeasure=false)
        f = WaterLily.pressure_force(sim) ./ (0.5 * sim.L * sim.U^2)
        push!(drag, f[1])
        push!(lift, f[2])
    end
    ts, drag, lift
end

time_wl, drag_wl, lift_wl = get_force_history!(wl; t_end=50.0)
time_lp, drag_lp, lift_lp = get_force_history!(lp; t_end=50.0)

pf = plot(time_wl, drag_wl, label="drag WL", xlabel="tU/L", ylabel="Pressure force coeff")
plot!(pf, time_lp, drag_lp, label="drag LP", ls=:dash)
plot!(pf, time_wl, lift_wl, label="lift WL")
plot!(pf, time_lp, lift_lp, label="lift LP", ls=:dash)
savefig(pf, joinpath(out_dir, "circle_force_compare.png"))

# Settled-regime metrics over t=25-50.
# Use absolute error for lift mean (reference is near zero).
mask_wl = time_wl .>= 25.0
mask_lp = time_lp .>= 25.0

lift_mean_abs_err = mean(lift_wl[mask_wl]) - mean(lift_lp[mask_lp])
drag_mean_err = 1 - mean(drag_lp[mask_lp]) / mean(drag_wl[mask_wl])
drag_std_err  = 1 - std(drag_lp[mask_lp])  / std(drag_wl[mask_wl])
lift_std_err  = 1 - std(lift_lp[mask_lp])  / std(lift_wl[mask_wl])

println("Lift mean abs error WL - LP:  ", lift_mean_abs_err)
println("Drag mean relative error (1 - LP/WL):  ", drag_mean_err)
println("Drag std  relative error (1 - LP/WL):  ", drag_std_err)
println("Lift std  relative error (1 - LP/WL):  ", lift_std_err)

# Pressure field comparison at t=50
p_wl = Array(wl.flow.p)
p_lp = Array(lp.flow.p)
clims = extrema([p_wl; p_lp])
pp1 = heatmap(p_wl', title="WaterLily pressure (t=50)", xlabel="x", ylabel="y",
              clims=clims, color=:RdBu)
pp2 = heatmap(p_lp', title="LilyPad pressure (t=50)", xlabel="x", ylabel="y",
              clims=clims, color=:RdBu)
pp = plot(pp1, pp2, layout=(1, 2), size=(1100, 420))
savefig(pp, joinpath(out_dir, "circle_pressure_final_compare.png"))
println("=== Pressure comparison at t=50 saved ===")
