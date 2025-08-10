using CUDA
using DINCAE
using DINCAE_utils
using Dates
using NCDatasets

# longitude range (east, west)
lon_range = [-7, -0.8]
# latitude range (south, north)
lat_range = [33.8, 38.2]
# time range (start, end)
time_range = [DateTime(2000,2,25), DateTime(2020,12,31)]


# local directory
localdir = expanduser("./data/")
# create directory
mkpath(localdir)
# filename of the subset
fname_subset = joinpath(localdir,"dataset.nc")
# filename of the clean data
fname = joinpath(localdir,"modis_cleanup.nc")
# filename of the data with added clouds for cross-validation
fname_cv = joinpath(localdir,"modis_cleanup_add_clouds.nc")
varname = "sst"

# Results of DINCAE will be placed in a sub-directory under `localdir`

outdir = joinpath(localdir,"Results")
mkpath(outdir)



# Load the NetCDF variable `sst` and `qual_sst`

ds = NCDataset(fname_subset)
sst = ds["sst"][:,:,:];
qual = ds["qual_sst"][:,:,:];

# We ignore all data points with missing quality flags,
# quality indicator exceeding 3 and temperature
# higher than 40°C.

sst_t = copy(sst)
sst_t[(qual .> 3) .& .!ismissing.(qual)] .= missing
sst_t[(sst_t .> 40) .& .!ismissing.(sst_t)] .= missing

@info "number of missing observations: $(count(ismissing,sst_t))"
@info "number of valid observations: $(count(.!ismissing,sst_t))"

# Clean-up the data to write them to disk.

varname = "sst"
fname = joinpath(localdir,"modis_cleanup.nc")
ds2 = NCDataset(fname,"c")
write(ds2,ds,exclude = ["sst","qual"])
defVar(ds2,varname,sst_t,("lon","lat","time"))
close(ds2)

# Add a land-sea mask to the file. Grid points with less than 5% of
# valid data are considered as land.

DINCAE_utils.add_mask(fname,varname; minseafrac = 0.05)

# Choose cross-validation points by adding clouds to the cleanest
# images (copied from the cloudiest images). This function will generate
# a file `fname_cv`.

DINCAE_utils.addcvpoint(fname,varname; mincvfrac = 0.10);

# # Reconstruct missing data
#
#
# F is the floating point number type for the neural network. Here we use
# single precision.

const F = Float32

# Test if CUDA is functional to use the GPU, otherwise the CPU is used.

if CUDA.functional()
    Atype = CuArray{F}
else
    @warn "No supported GPU found. We will use the CPU which is very slow. Please check https://developer.nvidia.com/cuda-gpus"
    Atype = Array{F}
end

# Setting the parameters of neural network.
# See the documentation of `DINCAE.reconstruct` for more information.

epochs = 1000
batch_size = 32
enc_nfilter_internal = [16, 30, 58, 110, 209]
clip_grad = 5.0
regularization_L2_beta = 0
ntime_win = 3
upsampling_method = :nearest
loss_weights_refine = (0.3,0.7)
save_epochs = 200:10:epochs


data = [
   (filename = fname_cv,
    varname = varname,
    obs_err_std = 1,
    jitter_std = 0.05,
    isoutput = true,
   )
]
data_test = data;
fnames_rec = [joinpath(outdir,"data-avg.nc")]
data_all = [data,data_test]



modeldir = joinpath(outdir, "Weights")
mkpath(modeldir)
# Start the training and reconstruction of the neural network.

start = time()
loss = DINCAE.reconstruct(
    Atype,data_all,fnames_rec;
    epochs = epochs,
    batch_size = batch_size,
    enc_nfilter_internal = enc_nfilter_internal,
    clip_grad = clip_grad,
    save_epochs = save_epochs,
    upsampling_method = upsampling_method,
    loss_weights_refine = loss_weights_refine,
    ntime_win = ntime_win,
    modeldir = modeldir
)
@info "Elapsed time is: $(time() - start)\n"

open(joinpath(outdir, "loss.txt"), "w") do io
    for l in loss
        println(io, l)
    end
end
println("Loss values saved to $(joinpath(outdir, "loss.txt"))")

# # Post process results
#
# Compute the RMS (Root Mean Squared error) with the independent validation data

case = (
    fname_orig = fname,
    fname_cv = fname_cv,
    varname = varname,
)
fnameavg = joinpath(outdir,"data-avg.nc")
cvrms = DINCAE_utils.cvrms(case,fnameavg)
@info "Cross-validation RMS error is: $cvrms"