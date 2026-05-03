using LilyPad
using WaterLily
using Test
using StaticArrays

check_compiler(compiler, parse_str) = try occursin(parse_str, read(`$compiler --version`, String)) catch _ false end
_cuda = check_compiler("nvcc", "release")
_cuda && using CUDA

function setup_backends()
    arrays = [Array]
    _cuda && CUDA.functional() && push!(arrays, CUDA.CuArray)
    return arrays
end

arrays = setup_backends()
include("maintests.jl")
