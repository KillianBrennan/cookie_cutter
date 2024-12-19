#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
---------------------------------------------------------
extract inflow from cookies
---------------------------------------------------------
IN

---------------------------------------------------------
OUT

---------------------------------------------------------
EXAMPLE CALL

python /home/kbrennan/cookie_cutter/extract_inflow.py /home/kbrennan/phd/data/climate/cookies/present /home/kbrennan/phd/data/climate/inflow/present

python /home/kbrennan/cookie_cutter/extract_inflow.py /home/kbrennan/phd/data/climate/cookies/future /home/kbrennan/phd/data/climate/inflow/future

---------------------------------------------------------
Killian P. Brennan
09.07.2024
---------------------------------------------------------
"""

import os
import argparse
import xarray as xr
import multiprocessing as mp


def main(cookie_dir, inflow_dir,x_slice=(-25,25),y_slice=(10,60)):
    subdomains = os.listdir(os.path.join(cookie_dir, 'subdomains'))

    # extractor(cookie_dir, inflow_dir, 'BI', x_slice, y_slice)
    # for dom in subdomains:
    #     print(dom)
    #     extractor(cookie_dir, inflow_dir, dom, x_slice, y_slice)
    
    # parallel
    pool = mp.Pool(len(subdomains))
    pool.starmap(extractor, [(cookie_dir, inflow_dir, dom, x_slice, y_slice) for dom in subdomains])

    return

def extractor(cookie_dir, inflow_dir, dom, x_slice, y_slice):
    files = os.listdir(os.path.join(cookie_dir, 'subdomains', dom))
    for i,file in enumerate(files):
        print(file)
        df = xr.open_dataset(os.path.join(cookie_dir, 'subdomains', dom, file))
        inflow = df.sel(x=slice(*x_slice), y=slice(*y_slice))
        center = df.sel(x=slice(*x_slice), y=slice(*x_slice))

        # compute statistics
        inflow_mean = inflow.mean(dim=['x', 'y'], skipna=True)
        inflow_max = inflow.max(dim=['x', 'y'], skipna=True)
        inflow_min = inflow.min(dim=['x', 'y'], skipna=True)

        center_mean = center.mean(dim=['x', 'y'], skipna=True)
        center_max = center.max(dim=['x', 'y'], skipna=True)
        center_min = center.min(dim=['x', 'y'], skipna=True)

        # add _mean, _max, _min to the variable names
        inflow_mean = inflow_mean.rename({v: v + '_inflow_mean' for v in inflow_mean.data_vars})
        inflow_max = inflow_max.rename({v: v + '_inflow_max' for v in inflow_max.data_vars})
        inflow_min = inflow_min.rename({v: v + '_inflow_min' for v in inflow_min.data_vars})

        center_mean = center_mean.rename({v: v + '_storm_mean' for v in center_mean.data_vars})
        center_max = center_max.rename({v: v + '_storm_max' for v in center_max.data_vars})
        center_min = center_min.rename({v: v + '_storm_min' for v in center_min.data_vars})

        # merge
        inflow_stats = xr.merge([inflow_mean, inflow_max, inflow_min])
        center_stats = xr.merge([center_mean, center_max, center_min])

        # merge inflow and center
        stats = xr.merge([inflow_stats, center_stats])

        if i == 0:
            # create output
            output = stats
        else:
            # concatenate to output along cookie_id dimension
            output = xr.concat([output, stats], dim='cookie_id')

    # save output
    output.to_netcdf(os.path.join(inflow_dir, dom + '.nc'))
    return
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='extract inflow from cookies')
    parser.add_argument('cookie_dir', type=str, help='directory of cookies')
    parser.add_argument('inflow_dir', type=str, help='directory of inflow')
    args = parser.parse_args()

    main(args.cookie_dir, args.inflow_dir)