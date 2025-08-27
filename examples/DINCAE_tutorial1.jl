# # Tutorial on DINCAE
#
# This script/notebook reconstructs missing data in satellite data using a neural
# network architecture called Convolutional Auto-Encoder described in the
# following articles:
#
# * Barth, A., Alvera-Azcárate, A., Licer, M., and Beckers, J.-M.: DINCAE 1.0: a convolutional neural network with error estimates to reconstruct sea surface temperature satellite observations, Geosci. Model Dev., 13, 1609–1622, https://doi.org/10.5194/gmd-13-1609-2020, 2020.
# * Barth, A., Alvera-Azcárate, A., Troupin, C., & Beckers, J.-M. (2022). DINCAE 2.0: multivariate convolutional neural network with error estimates to reconstruct sea surface temperature satellite and altimetry observations. Geoscientific Model Development, 15(5), 2183–2196. https://doi.org/10.5194/gmd-15-2183-2022
#
# The example here uses MODIS sea surface temperature from the Physical
# Oceanography Distributed Active Archive Center (PO.DAAC) JPL, NASA.
# More information is available at https://dx.doi.org/10.5067/MODST-1D4D9
#
# This notebook/script is intended to be run on a GPU with CUDA support (NVIDIA GPU)
# with a least 8 GB of RAM.

# The modules necessary to run the tutorial can be installed with:
#
# ```julia
# using Pkg
# Pkg.activate("/file/path/") # needs to be repeated for every session
# Pkg.instantiate() # install all package
# ```
#
# where `"/file/path/"` is the path containing this file and the file `Project.toml`.
# *Alternatively*, one can also install the modules in the default environment with:
#
# ```julia
# using Pkg
# Pkg.add(["CUDA","NCDatasets","PyPlot"])
# ```
#
# See also https://github.com/gher-uliege/DINCAE.jl#installation
#
# Load the necessary modules

using CUDA
using DINCAE
using DINCAE_utils
using Dates
using NCDatasets

# ## Data download
#
# First we define the domain and time interval.
#
# The example is tested here for a short time frame but for realistic
# applications one should use a much longer time-range (like 10, 20 years or more)

# longitude range (east, west)
lon_range = [-7, -0.8]
# latitude range (south, north)
lat_range = [33.8, 38.2]
# time range (start, end)
time_range = [DateTime(2000,2,25), DateTime(2020,12,31)]
#time_range = [DateTime(2000,2,25), DateTime(2000,3,31)]
#time_range = [DateTime(2001,1,1), DateTime(2001,12,31)]


# local directory
localdir = expanduser("./data/")
# create directory
mkpath(localdir)
# filename of the subset
fname_subset = joinpath(localdir,"modis_subset.nc")
# filename of the clean data
fname = joinpath(localdir,"modis_cleanup.nc")
# filename of the data with added clouds for cross-validation
fname_cv = joinpath(localdir,"modis_cleanup_add_clouds.nc")
varname = "sst"

# Results of DINCAE will be placed in a sub-directory under `localdir`

outdir = joinpath(localdir,"Results")
mkpath(outdir)

# The variable `url` is the OPeNDAP data URL of the MODIS data. Note the final
# `#fillmismatch` (look here https://github.com/Unidata/netcdf-c/issues/1299#issuecomment-458312804
# for `#fillmismatch` the suffix).
# The downloading can take several minutes.

#  `Error { code = 500; message = "Java heap space"; }`

if !isfile(fname_subset)
    download("https://dox.ulg.ac.be/index.php/s/ckHBdhDzAKERwPb/download",fname_subset)
end

# ```julia
# url = "https://thredds.jpl.nasa.gov/thredds/dodsC/ncml_aggregation/OceanTemperature/modis/terra/11um/4km/aggregate__MODIS_TERRA_L3_SST_THERMAL_DAILY_4KM_DAYTIME_V2019.0.ncml#fillmismatch"
# ds = NCDataset(url)
# # find indices withing the longitude, latitude and time range
# i = findall(lon_range[1] .<= ds["lon"][:] .<= lon_range[end]);
# j = findall(lat_range[1] .<= ds["lat"][:] .<= lat_range[end]);
# n = findall(time_range[1] .<= ds["time"][:] .<= time_range[end]);
# # Write subset to disk
# write(fname_subset,ds,idimensions = Dict(
#    "lon" => i,
#    "lat" => j,
#    "time" => n))
# close(ds)
# @info "NetCDF subset ($(length(n)) slices) written $fname_subset"
# ```

# ## Data preparation
#
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
sst_t[(sst_t .<= 0) .& .!ismissing.(sst_t)] .= missing # Added by me

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
# enc_nfilter_internal = round.(Int,32 * 2 .^ (0:4))
clip_grad = 5.0
# regularization_L2_beta = 0
ntime_win = 3
upsampling_method = :nearest
loss_weights_refine = (0.3,0.7)
save_epochs = 200:10:epochs

# Paper Params
initial_lr = 0.00058
enc_nfilter_internal = [16, 30, 58, 110, 209]
regularization_L2_beta = 0.0001


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

# Use these parameters for a quick test:

# epochs = 10
# save_epochs = epochs:epochs

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
    learning_rate = initial_lr
)

elapsed_seconds = time() - start
@info "Elapsed time is: $(elapsed_seconds) seconds"

open(joinpath(outdir, "loss.txt"), "w") do io
    for l in loss
        println(io, l)
    end
end
println("Loss values saved to $(joinpath(outdir, "loss.txt"))")

# Plot the loss function

# plot(loss)
# ylim(extrema(loss[2:end]))
# xlabel("epochs")
# ylabel("loss");

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

# Next we plot all time instances. The figures will be placed in the
# directory `figdir`

# figdir = joinpath(outdir,"Fig")
# DINCAE_utils.plotres(case,fnameavg, clim = nothing, figdir = figdir,
#                      clim_quantile = (0.01,0.99),
#                      which_plot = :cv)
# @info "Figures are in $(figdir)"


# Example reconstruction for 2001-09-12
# ![reconstruction for the 2001-09-12](Fig/data-avg_2001-09-12.png)
# Panel (a) is the original data where we have added clouds (panel (b)). The
# reconstuction based on the data in panel (b) is shown in panel (c) together
# with its expected standard deviation error (panel (d)).