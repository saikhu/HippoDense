using Flux3D, Flux, Makie, CUDA
using Flux: onehotbatch, onecold, onehot, crossentropy
using Statistics: mean
using Base.Iterators: partition

Makie.inline!(false)
Makie.set_theme!(show_axis = false)


batch_size = 32
lr = 3e-4
epochs = 5
num_classes = 10 #possible values {10,40}
numPoints = 1024

dset = ModelNet10.dataset(;
    npoints = numPoints,
    transform = NormalizePointCloud(),
)
val_dset = ModelNet10.dataset(;
    train = false,
    npoints = numPoints,
    transform = NormalizePointCloud(),
)

























































