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

python /home/kbrennan/cookie_cutter/calculate_composites.py /home/kbrennan/phd/data/climate/cookies/future --filter_lifetime 150 --filter_w 25

---------------------------------------------------------
Killian P. Brennan
08.05.2024
---------------------------------------------------------
"""

import os
import sys

import xarray as xr
import numpy as np

import argparse

import multiprocessing as mp


def main(
    cookie_dir, filter_lifetime=None, filter_diameter=None, filter_w=None, backend="cdo"
):
    composite_dir = os.path.join(cookie_dir, "composites_filtered")
    os.makedirs(composite_dir, exist_ok=True)
    # # remove existing composites
    # for f in os.listdir(composite_dir):
    #     os.remove(os.path.join(composite_dir, f))

    subdomains_dir = os.path.join(cookie_dir, "subdomains")

    subdomains = os.listdir(subdomains_dir)
    # must be directories
    subdomains = [
        sub for sub in subdomains if os.path.isdir(os.path.join(subdomains_dir, sub))
    ]
    do_calculations_cdo(cookie_dir, composite_dir, "BI")

    print("subdomains", subdomains)
    # do calculations for subdomains in parallel
    with mp.Pool(len(subdomains)) as pool:
        if backend == "cdo":
            pool.starmap(
                do_calculations_cdo,
                [
                    (
                        cookie_dir,
                        composite_dir,
                        dom,
                        filter_lifetime,
                        filter_diameter,
                        filter_w,
                    )
                    for dom in subdomains
                ],
            )
        elif backend == "xarray":
            pool.starmap(
                do_calculations,
                [
                    (cookie_dir, composite_dir, dom, filter_lifetime, filter_diameter)
                    for dom in subdomains
                ],
            )
    return


def do_calculations_cdo(
    cookie_dir,
    composite_dir,
    dom,
    filter_lifetime=None,
    filter_diameter=None,
    filter_w=None,
):
    subdomains_dir = os.path.join(cookie_dir, "subdomains")

    # mean
    writedir = construct_writedir(
        composite_dir, dom, filter_lifetime, filter_diameter, stat="mean"
    )
    cdo_command = (
        f"cdo ensmean {os.path.join(subdomains_dir, dom, 'cookie_*.nc')} {writedir}"
    )
    # std
    writedir = construct_writedir(
        composite_dir, dom, filter_lifetime, filter_diameter, stat="std"
    )
    cdo_command = (
        f"cdo ensstd {os.path.join(subdomains_dir, dom, 'cookie_*.nc')} {writedir}"
    )
    # q90
    writedir = construct_writedir(
        composite_dir, dom, filter_lifetime, filter_diameter, stat="q90"
    )
    cdo_command = (
        f"cdo enspctl,90 {os.path.join(subdomains_dir, dom, 'cookie_*.nc')} {writedir}"
    )

    os.system(cdo_command)

    return


def do_calculations(
    cookie_dir,
    composite_dir,
    dom,
    filter_lifetime=None,
    filter_diameter=None,
    filter_w=None,
):
    subdomains_dir = os.path.join(cookie_dir, "subdomains")
    cookies = load_cookies(os.path.join(subdomains_dir, dom))
    cookies = filter_cookies(cookies, filter_lifetime, filter_diameter)
    composite = calculate_composite(cookies)
    composite = composite.expand_dims({"domain": [dom]})
    writedir = construct_writedir(
        composite_dir, dom, filter_lifetime, filter_diameter, stat="comp"
    )
    composite.to_netcdf(writedir)
    print(f"finished composite for {dom}")
    return


def construct_writedir(
    composite_dir,
    dom,
    filter_lifetime=None,
    filter_diameter=None,
    filter_w=None,
    stat=None,
):
    filter_str = ""
    if filter_lifetime is not None:
        filter_str += f"_lifetime{filter_lifetime}"
    if filter_diameter is not None:
        filter_str += f"_max_val{filter_diameter}"
    if filter_w is not None:
        filter_str += f"_w{filter_w}"
    stat = "_" + stat if stat else ""
    writedir = os.path.join(composite_dir, dom + filter_str + stat + ".nc")
    return writedir


def filter_cookies(cookies, filter_lifetime=None, filter_diameter=None, filter_w=None):
    if filter_lifetime is not None:
        cookies = cookies.where(
            cookies["cell_lifespan"] >= np.timedelta64(filter_lifetime, "min"),
            drop=True,
        )
    if filter_diameter is not None:
        cookies = cookies.where(cookies["max_val"] >= filter_diameter, drop=True)
    if filter_w is not None:
        cookies = cookies.where(
            cookies.sel(presssure=400).W.max() >= filter_w, drop=True
        )
    return cookies


def load_cookies(cookie_dir):
    cookies = xr.open_mfdataset(
        os.path.join(cookie_dir, "cookie_*.nc"),
        combine="by_coords",
        parallel=True,
        chunks={"cookie_id": 1000},
    )
    cookies = add_season_to_cookies(cookies)

    cookies = cookies.drop(
        ["real_time", "t_rel_start", "t_rel_end", "t_rel_max", "cell_lifespan", "itime"]
    )

    return cookies


def add_season_to_cookies(cookies):
    # add season to cookies (DJF, MAM, JJA, SON)
    cookies["season"] = xr.where(
        cookies["real_time.month"].isin([12, 1, 2]),
        "DJF",
        xr.where(
            cookies["real_time.month"].isin([3, 4, 5]),
            "MAM",
            xr.where(cookies["real_time.month"].isin([6, 7, 8]), "JJA", "SON"),
        ),
    )
    return cookies


def calculate_composite(cookies):
    n_fields = cookies.groupby("season").count(dim="cookie_id").T
    n_cookies = cookies.groupby("season").count(dim="cookie_id").max_val

    # todo: this should fix the problem with only having one season after .groupby, but it doesn't
    # is not a problem, as long as all subdomains have cookies in at least two seasons.
    # if "season" not in n_fields.dims:
    #     n_fields = n_fields.expand_dims("season")
    #     n_cookies = n_cookies.expand_dims("season")

    mean = cookies.groupby("season").mean(dim="cookie_id", skipna=True, keep_attrs=True)
    mean = mean.compute()
    # std = cookies.groupby("season").std(dim="cookie_id", skipna=True, keep_attrs=True)
    # std = std.compute()
    q90 = cookies.groupby("season").quantile(
        0.9, dim="cookie_id", skipna=True, keep_attrs=True
    )
    q90 = q90.compute()
    # q98 = cookies.groupby("season").quantile(
    #     0.98, dim="cookie_id", skipna=True, keep_attrs=True
    # )
    # q98 = q98.compute()

    composite = xr.Dataset()
    for var in mean.data_vars:
        composite[var] = mean[var]
        # composite[var + "_std"] = std[var]
        composite[var + "_q90"] = q90[var]
        # composite[var + "_q98"] = q98[var]

    composite.attrs = mean.attrs

    composite.attrs["compositing"] = (
        "mean is applied along cookie_id axis and saved to each variable as original name, standard deviation is saved to the _std ending variable names"
    )

    composite["n_cookies"] = n_cookies
    composite["n_cookies"].attrs = {"long_name": "number of cookies in composite"}
    composite["n_fields"] = n_fields
    composite["n_fields"].attrs = {"long_name": "number of fields in composite"}

    seasons = ["DJF", "MAM", "JJA", "SON"]
    # add empty nan filled composite if season not present
    for season in seasons:
        if season not in composite["season"].values:
            dummy = xr.full_like(composite.isel(season=0), np.nan)
            dummy["season"] = season
            composite = xr.concat([composite, dummy], dim="season")

    return composite


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate composites from cookies")
    parser.add_argument("cookie_dir", type=str, help="Directory containing cookies")
    parser.add_argument(
        "--filter_lifetime",
        type=int,
        default=None,
        help="Filter out cookies with a lifetime less than this value (minutes)",
    )
    parser.add_argument(
        "--filter_diameter",
        type=int,
        default=None,
        help="Filter out cookies with a maximum hail diameter less than this value (mm)",
    )
    parser.add_argument(
        "--filter_w",
        type=int,
        default=None,
        help="Filter out cookies with a maximum updraft velocity less than this value (m/s)",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="cdo",
        help="Use xarray or cdo for calculations",
    )

    args = parser.parse_args()
    main(
        args.cookie_dir,
        args.filter_lifetime,
        args.filter_diameter,
        args.filter_w,
        args.backend,
    )
