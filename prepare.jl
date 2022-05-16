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

dset = ModelNet10.dataset(;
    mode = :pointcloud,
    npoints = npoints,
    transform = NormalizePointCloud(),
)
val_dset = ModelNet10.dataset(;
    mode = :pointcloud,
    train = false,
    npoints = npoints,
    transform = NormalizePointCloud(),
)


visualize(dset[11], markersize = 0.1)
data = [dset[i].data.points for i = 1:length(dset)]
labels =
    onehotbatch([dset[i].ground_truth for i = 1:length(dset)], 1:num_classes)

valX = cat([val_dset[i].data.points for i = 1:length(val_dset)]..., dims = 3)
valY = onehotbatch(
    [val_dset[i].ground_truth for i = 1:length(val_dset)],
    1:num_classes,
)

TRAIN = [
    (cat(data[i]..., dims = 3), labels[:, i])
    for i in partition(1:length(data), batch_size)
]
VAL = (valX, valY)

m = PointNet(num_classes)

loss(x, y) = crossentropy(m(x), y)
accuracy(x, y) =
    mean(onecold(cpu(m(x)), 1:num_classes) .== onecold(cpu(y), 1:num_classes))

# Defining learning rate and optimizer
opt = Flux.ADAM(lr)

# Using GPU for fast training [Optional]
# We can convert the 3D model to GPU or CPU usinggpu and cpu, and also changing the dataloader using same function

m = m |> gpu
TRAIN = TRAIN |> gpu
VAL = VAL |> gpu

# Training the 3D model
ps = params(m)
for epoch = 1:epochs
    running_loss = 0
    for d in TRAIN
        gs = gradient(ps) do
            training_loss = loss(d...)
            running_loss += training_loss
            return training_loss
        end
        Flux.update!(opt, ps, gs)
    end
    print("Epoch: $(epoch), epoch_loss: $(running_loss), accuracy: $(accuracy(VAL...))\n")
end
@show accuracy(VAL...)
