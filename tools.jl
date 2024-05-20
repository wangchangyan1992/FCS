using PyCall
using LaTeXStrings
using Chain: @chain

len(arr)              = length(arr)
Base.:∋(vec, x)       = push!(vec, x)
Base.:∋(vec, x...)    = push!(vec, x...)
