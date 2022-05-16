using CUDA
using Flux

# to check the GPU and CUDA information
# CUDA.versioninfo()
# CUDA.functional()

# To disable all devices:
# $ export CUDA_VISIBLE_DEVICES='-1'

# To select specific devices by device id:
# $ export CUDA_VISIBLE_DEVICES='0,1'

W = cu(rand(2, 5)) # a 2×5 CuArray
b = cu(rand(2))

predict(x) = W*x .+ b
loss(x, y) = sum((predict(x) .- y).^2)

x, y = cu(rand(5)), cu(rand(2)) # Dummy data
loss(x, y) # ~ 3


d = Dense(10 => 5, σ)
d = fmap(cu, d)
d.weight # CuArray
d(cu(rand(10))) # CuArray output

m = Chain(Dense(10 => 5, σ), Dense(5 => 2), softmax)
m = fmap(cu, m)
d(cu(rand(10)))

