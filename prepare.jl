using Flux3D, Flux, Makie, CUDA
using Flux: onehotbatch, onecold, onehot, crossentropy
using Statistics: mean
using Base.Iterators: partition

# to check the GPU and CUDA information
# CUDA.versioninfo()
# CUDA.functional()

# To disable all devices:
# $ export CUDA_VISIBLE_DEVICES='-1'

# To select specific devices by device id:
# $ export CUDA_VISIBLE_DEVICES='0,1'

Makie.inline!(false)
Makie.set_theme!(show_axis = false)

