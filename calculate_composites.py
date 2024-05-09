#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
---------------------------------------------------------
calculate composites from cookies
---------------------------------------------------------
IN

---------------------------------------------------------
OUT

---------------------------------------------------------
EXAMPLE CALL

python /home/kbrennan/cookie_cutter/calculate_composites.py /home/kbrennan/phd/data/climate/cookies/future

---------------------------------------------------------
Killian P. Brennan
08.05.2024
---------------------------------------------------------
"""

import os

import xarray as xr
import numpy as np

import argparse

import multiprocessing as mp


def main(cookie_dir):
    composite_dir = os.path.join(cookie_dir, "composites")
    os.makedirs(composite_dir, exist_ok=True)
    # remove existing composites
    for f in os.listdir(composite_dir):
        os.remove(os.path.join(composite_dir, f))

    subdomains_dir = os.path.join(cookie_dir, "subdomains")

    subdomains = os.listdir(subdomains_dir)
    # must be directories
    subdomains = [
        sub for sub in subdomains if os.path.isdir(os.path.join(subdomains_dir, sub))
    ]
    # do_calculations(cookie_dir, composite_dir, "BI")

    print("subdomains", subdomains)
    # do calculations for subdomains in parallel
    with mp.Pool(len(subdomains)) as pool:
        pool.starmap(
            do_calculations, [(cookie_dir, composite_dir, dom) for dom in subdomains]
        )
    return


def do_calculations(cookie_dir, composite_dir, dom):
    subdomains_dir = os.path.join(cookie_dir, "subdomains")
    cookies = load_cookies(os.path.join(subdomains_dir, dom))
    composite = calculate_composite(cookies)
    composite = composite.expand_dims({"domain": [dom]})
    composite.to_netcdf(os.path.join(composite_dir, f"{dom}_comp.nc"))
    print(f"finished composite for {dom}")
    return


def load_cookies(cookie_dir):
    cookies = xr.open_mfdataset(
        os.path.join(cookie_dir, "*.nc"),
        combine="by_coords",
        parallel=True,
        chunks={"cookie_id": 1000},
    )
    # add season to cookies (DJF, MAM, JJA, SON)
    cookies["season"] = xr.where(
        cookies["real_time.month"].isin([12, 1, 2]),
        "DJF",
        xr.where(
            cookies["real_time.month"].isin([3, 4, 5]),
            "MAM",
            xr.where(
                cookies["real_time.month"].isin([6, 7, 8]),
                "JJA",
                xr.where(cookies["real_time.month"].isin([9, 10, 11]), "SON", None),
            ),
        ),
    )

    return cookies


def calculate_composite(cookies):
    cookies = cookies.drop(
        ["real_time", "t_rel_start", "t_rel_end", "t_rel_max", "cell_lifespan"]
    )
    n_fields = cookies.groupby("season").count(dim="cookie_id").T
    n_cookies = cookies.groupby("season").count(dim="cookie_id").max_val

    mean = cookies.groupby("season").mean(dim="cookie_id", skipna=True, keep_attrs=True)
    mean = mean.compute()
    std = cookies.groupby("season").std(dim="cookie_id", skipna=True, keep_attrs=True)
    std = std.compute()

    composite = xr.Dataset()
    for var in mean.data_vars:
        composite[var] = mean[var]
        composite[var + "_std"] = std[var]

    composite.attrs = mean.attrs

    composite.attrs["compositing"] = (
        "mean is applied along cookie_id axis and saved to each variable as original name, standard deviation is saved to the _std ending variable names"
    )

    composite["n_cookies"] = n_cookies
    composite["n_cookies"].attrs = {"long_name": "number of cookies in composite"}
    composite["n_fields"] = n_fields
    composite["n_fields"].attrs = {"long_name": "number of fields in composite"}
    return composite


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate composites from cookies")
    parser.add_argument("cookie_dir", type=str, help="Directory containing cookies")
    args = parser.parse_args()
    main(args.cookie_dir)
