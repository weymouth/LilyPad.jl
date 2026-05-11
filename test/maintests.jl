import LilyPad: scalar_advect!, sim_step!, quad, LilyFlow

@testset "util.jl" begin
    T = Float32

    # Linear interpolation wrapper clamps safely out-of-bounds.
    a2 = fill(T(3.7), 8, 8)
    x_out2 = SVector{2,T}(-10, 100)
    @test WaterLily.interp(x_out2, a2) ≈ T(3.7)

    # Quadratic interpolation wrapper also clamps safely.
    @test LilyPad.quad(x_out2, a2) ≈ T(3.7)

    # 2D quadratic exactness: tensor-product quadratic interpolation should
    # exactly reproduce any degree-2 polynomial field on a uniform grid.
    N2 = 10
    p2(x, y) = T(0.7) * x^2 + T(0.3) * y^2 + T(0.2) * x * y + T(1.1) * x - T(0.8) * y + T(0.4)
    f2 = zeros(T, N2, N2)
    for i in 1:N2, j in 1:N2
        x = T(i) - T(1.5)
        y = T(j) - T(1.5)
        f2[i, j] = p2(x, y)
    end
    xq2 = SVector{2,T}(T(4.2), T(5.1))
    @test LilyPad.quad(xq2, f2) ≈ p2(xq2[1], xq2[2]) atol=1f-5

end

@testset "SemiAdvect.jl" begin
    D, T = 2, Float32
    N = (6, 6)  # interior cells; full array is (N.+2) with ghosts
    U = SVector{D,T}(1, 2)
    dt = T(0.3)

    u_cpu = zeros(T, N .+ 2..., D); apply!((i, x) -> U[i], u_cpu)
    x0 = SVector{D,T}(1.0, 1.5)
    @test LilyPad.departure(x0, u_cpu, dt) ≈ x0 - dt * U

    for f in arrays
        # 1) Linear field transport using scalar_advect! for both components.
        u = zeros(T, N .+ 2..., D) |> f
        apply!((i, x) -> U[i], u)
        src = zeros(T, N .+ 2..., D) |> f
        apply!((i, x) -> x[i], src)
        out = zeros(T, N .+ 2..., D) |> f

        for i in 1:D
            scalar_advect!(selectdim(out, D + 1, i), selectdim(src, D + 1, i), u, dt, i)
        end

        ref = zeros(T, N .+ 2..., D)
        apply!((i, x) -> x[i] - dt * U[i], ref)
        diff = Array(out) - ref
        sz = size(out)[1:D]
        for i in 1:D
            r_first = ntuple(k -> k == i ? (3:3) : (2:sz[k]-1), D)
            r_last  = ntuple(k -> k == i ? (sz[k]-1:sz[k]-1) : (2:sz[k]-1), D)
            @test maximum(abs.(diff[r_first..., i])) < 1f-4
            @test maximum(abs.(diff[r_last..., i])) < 1f-4
        end

        # 2) Nontrivial field transported along lower boundary for i=1.
        p(x, y) = 0.3f0 * x^2 + 0.2f0 * y^2 + 0.15f0 * x * y + 0.4f0 * x - 0.7f0 * y + 1.2f0
        Nb = 20
        u_along = zeros(T, Nb + 2, Nb + 2, D) |> f
        apply!((i, x) -> i == 1 ? 1f0 : 0f0, u_along)
        src_along = zeros(T, Nb + 2, Nb + 2, D) |> f
        apply!((i, x) -> p(x[1], x[2]), src_along)
        adv_along = zeros(T, Nb + 2, Nb + 2) |> f
        scalar_advect!(adv_along, selectdim(src_along, 3, 1), u_along, T(1), 1)

        ref_along = zeros(T, Nb + 2, Nb + 2, D)
        apply!((i, x) -> p(x[1] - T(1), x[2]), ref_along)
        d_along = Array(adv_along) - Array(selectdim(ref_along, 3, 1))
        r_low = (3:Nb+1, 2:2)
        @test maximum(abs.(d_along[r_low...])) < 2f-4

        # 3) Nontrivial field transported toward lower boundary for i=1.
        u_cross = zeros(T, Nb + 2, Nb + 2, D) |> f
        apply!((i, x) -> i == 2 ? -1f0 : 0f0, u_cross)
        src_cross = zeros(T, Nb + 2, Nb + 2, D) |> f
        apply!((i, x) -> p(x[1], x[2]), src_cross)
        adv_cross = zeros(T, Nb + 2, Nb + 2) |> f
        scalar_advect!(adv_cross, selectdim(src_cross, 3, 1), u_cross, T(1), 1)

        ref_cross = zeros(T, Nb + 2, Nb + 2, D)
        apply!((i, x) -> p(x[1], x[2] + T(1)), ref_cross)
        d_cross = Array(adv_cross) - Array(selectdim(ref_cross, 3, 1))
        @test maximum(abs.(d_cross[r_low...])) < 2f-4
        @test all(isfinite, Array(adv_cross))
    end
end

@testset "LilyFlow.jl" begin
    U = (2f0 / 3, -1f0 / 3)
    N = (2^4, 2^4)

    for mem in arrays
        flow = LilyFlow(N, U; mem, T=Float32)
        pois = MultiLevelPoisson(flow.p, flow.μ₀, flow.σ)

        WaterLily.mom_step!(flow, pois)

        # Uniform flow should stay close to a fixed point after one full step.
        @test L₂(flow.u[:, :, 1] .- U[1]) < 2f-5
        @test L₂(flow.u[:, :, 2] .- U[2]) < 1f-5

        # Projection should keep interior divergence small.
        @inside flow.σ[I] = WaterLily.div(I, flow.u)
        @test maximum(abs.(Array(flow.σ)[inside(flow.p)])) < 1f-3

        # Second step: velocity must stay near U.
        WaterLily.mom_step!(flow, pois)
        @test L₂(flow.u[:, :, 1] .- U[1]) < 2f-5
        @test L₂(flow.u[:, :, 2] .- U[2]) < 1f-5
    end
end

@testset "LilyPadSim.jl" begin
    U = (2f0 / 3, -1f0 / 3)
    N = (2^4, 2^4)
    L = N[1]

    for f in arrays
        sim = LilyPadSim(N, U, L; Δt=0.25, T=Float32, mem=f)
        ref = Simulation(N, U, L; Δt=0.25, T=Float32, mem=f)

        @test sim_time(sim) == 0

        sim_step!(sim)
        sim_step!(ref)

        @test sim_time(sim) > 0
        @test length(sim.flow.Δt) == 2
        @test L₂(sim.flow.u[:, :, 1] .- ref.flow.u[:, :, 1]) < 1f-4
        @test L₂(sim.flow.u[:, :, 2] .- ref.flow.u[:, :, 2]) < 1f-4

        # Second step: LP must stay close to WL.
        sim_step!(sim)
        sim_step!(ref)
        @test L₂(sim.flow.u[:, :, 1] .- ref.flow.u[:, :, 1]) < 1f-3
        @test L₂(sim.flow.u[:, :, 2] .- ref.flow.u[:, :, 2]) < 1f-3
    end

end
