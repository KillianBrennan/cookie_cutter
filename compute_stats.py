import os
import xarray as xr
import dask
from dask.distributed import Client

client = Client(n_workers=16)
@dask.delayed

def main():
    inpath = "/home/kbrennan/phd/data/climate/cookies/present"
    outpath = os.path.join(inpath, "stats")
    os.makedirs(outpath, exist_ok=True)

    # load data
    dset = xr.open_mfdataset(inpath + "/*.nc", combine="by_coords")

    # compute stats in palallel
    d_chunks = dask.array.from_array(dset, chunks=(1000, 1000))
    mean = d_chunks.mean(axis="cell_id").compute()

    # save stats
    mean.to_netcdf(os.path.join(outpath, "mean.nc"))
    return


if __name__ == "__main__":
    main()
