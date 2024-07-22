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
python /home/kbrennan/cookie_cutter/calculate_composites.py /home/kbrennan/phd/data/climate/cookies/future /home/kbrennan/phd/data/climate/grids/subdomains_lonlat.nc /home/kbrennan/phd/data/climate/composites/future/comp_n0.5 --filter_n 0.5

python /home/kbrennan/cookie_cutter/calculate_composites.py /home/kbrennan/phd/data/climate/cookies/present /home/kbrennan/phd/data/climate/grids/subdomains_lonlat.nc /home/kbrennan/phd/data/climate/composites/present/comp_n0.5 --filter_n 0.5
---
python /home/kbrennan/cookie_cutter/calculate_composites.py /home/kbrennan/phd/data/climate/cookies/future /home/kbrennan/phd/data/climate/grids/subdomains_lonlat.nc /home/kbrennan/phd/data/climate/composites/future/comp_p --filter_quantile 0.9

python /home/kbrennan/cookie_cutter/calculate_composites.py /home/kbrennan/phd/data/climate/cookies/present /home/kbrennan/phd/data/climate/grids/subdomains_lonlat.nc /home/kbrennan/phd/data/climate/composites/present/comp_p --filter_quantile 0.9
--
python /home/kbrennan/cookie_cutter/calculate_composites.py /home/kbrennan/phd/data/climate/cookies/future /home/kbrennan/phd/data/climate/grids/subdomains_lonlat.nc /home/kbrennan/phd/data/climate/composites/future/comp_f --filter_lifetime 150 --filter_w 25

python /home/kbrennan/cookie_cutter/calculate_composites.py /home/kbrennan/phd/data/climate/cookies/present /home/kbrennan/phd/data/climate/grids/subdomains_lonlat.nc /home/kbrennan/phd/data/climate/composites/present/comp_f --filter_lifetime 150 --filter_w 25
--
python /home/kbrennan/cookie_cutter/calculate_composites.py /home/kbrennan/phd/data/climate/cookies/future /home/kbrennan/phd/data/climate/grids/subdomains_lonlat.nc /home/kbrennan/phd/data/climate/composites/future/comp

python /home/kbrennan/cookie_cutter/calculate_composites.py /home/kbrennan/phd/data/climate/cookies/present /home/kbrennan/phd/data/climate/grids/subdomains_lonlat.nc /home/kbrennan/phd/data/climate/composites/present/comp
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
    cookie_dir,
    subdomain_dir,
    composite_dir,
    filter_lifetime=None,
    filter_diameter=None,
    filter_w=None,
    filter_quantile=None,
    filter_n=None,
    serial=False,
    n_bootstrap=100,
):
    os.makedirs(composite_dir, exist_ok=True)
    # # remove existing composites
    # for f in os.listdir(composite_dir):
    #     os.remove(os.path.join(composite_dir, f))

    subdomains_dir = os.path.join(cookie_dir, "subdomains")

    if not os.path.exists(subdomains_dir):
        print(
            f"subdomains directory {subdomains_dir} does not exist, did you run assign_subdomains.py?"
        )
        sys.exit()

    subdomains = os.listdir(subdomains_dir)
    # must be directories
    subdomains = [
        sub for sub in subdomains if os.path.isdir(os.path.join(subdomains_dir, sub))
    ]

    subdomains.remove('MDS')
    subdomains.remove('MDL')

    # do_calculations_cdo(cookie_dir, composite_dir, "MD", filter_lifetime, filter_diameter, filter_w)
    # exit()
    print("subdomains", subdomains)

    if serial:
        for dom in subdomains:
            do_calculations_cdo(
                cookie_dir,
                composite_dir,
                dom,
                subdomain_dir,
                filter_lifetime,
                filter_diameter,
                filter_w,
                filter_quantile,
                filter_n,
                n_bootstrap,
            )
    else:
        # do calculations for subdomains in parallel
        with mp.Pool(len(subdomains)) as pool:
            pool.starmap(
                do_calculations_cdo,
                [
                    (
                        cookie_dir,
                        composite_dir,
                        dom,
                        subdomain_dir,
                        filter_lifetime,
                        filter_diameter,
                        filter_w,
                        filter_quantile,
                        filter_n,
                        n_bootstrap,
                    )
                    for dom in subdomains
                ],
            )
    return


def do_calculations_cdo(
    cookie_dir,
    composite_dir,
    dom,
    subdomain_dir,
    filter_lifetime=None,
    filter_diameter=None,
    filter_w=None,
    filter_quantile=None,
    filter_n=None,
    n_bootstrap=10,
):

    # seasons = ["DJF", "MAM", "JJA", "SON", "YEAR"]
    # seasons = ["DJF", "MAM", "JJA", "SON"]
    seasons = ["MAM", "JJA", "SON"]
    # seasons = ['YEAR']
    months = {
        "DJF": "12,01,02",
        "MAM": "03,04,05",
        "JJA": "06,07,08",
        "SON": "09,10,11",
        "YEAR": "01,02,03,04,05,06,07,08,09,10,11,12",
    }

    all_cookies = os.listdir(os.path.join(cookie_dir, "subdomains", dom))
    all_cookies = filter_cookies_cdo(
        all_cookies,
        cookie_dir,
        dom,
        subdomain_dir,
        filter_lifetime,
        filter_diameter,
        filter_w,
        filter_quantile,
        filter_n,
    )
    for season in seasons:
        # cookies that are in the format cookie_YYYYMM*
        cookies_in_season = [c for c in all_cookies if c[11:13] in months[season]]
        cookies_in_season = [
            os.path.join(cookie_dir, "subdomains", dom, c) for c in cookies_in_season
        ]

        # make temporary directory
        temp_dir = os.path.join(composite_dir, "temp_" + dom + "_" + season)
        os.makedirs(temp_dir, exist_ok=True)

        # ---------------------------------------------------------
        # calculate bootstrap composites
        # ---------------------------------------------------------
        # resample 100 times with replacement and calculate the mean
        # from those means calculate mean, std, q25, q75, median, q99 q01 for each variable

        # make temporary directory for the bootstrapped means
        temp_dir_boot = os.path.join(temp_dir, "boot")
        os.makedirs(temp_dir_boot, exist_ok=True)
        # make temporary directory for the resampled cookies
        temp_dir_resampled = os.path.join(temp_dir, "resampled")
        os.makedirs(temp_dir_resampled, exist_ok=True)

        # calculate bootstrapped means
        for i in range(n_bootstrap):
            # resample with replacement
            resampled_cookies = np.random.choice(
                cookies_in_season, len(cookies_in_season), replace=True
            )
            # link resampled cookies to temp_dir_resampled (add str(j) to allow for dublicate samples)
            for j, c in enumerate(resampled_cookies):
                os.symlink(
                    c,
                    os.path.join(
                        temp_dir_resampled,
                        os.path.basename(c).split(".")[0] + f"_{j}.nc",
                    ),
                )
            # calculate mean
            cdo_command = (
                f"cdo -w -O ensmean  {temp_dir_resampled}/* {temp_dir_boot}/boot_{i}.nc"
            )
            os.system(cdo_command)

            # remove resampled cookies
            for c in os.listdir(temp_dir_resampled):
                os.remove(os.path.join(temp_dir_resampled, c))
        # calculate mean, std, q25, q75, median, q99, q01 from bootstrapped means
        # mean
        writedir = construct_writedir(
            composite_dir,
            dom,
            filter_lifetime,
            filter_diameter,
            filter_w,
            filter_quantile,
            filter_n,
            stat="boot_mean",
            season=season,
            n_cookies=len(cookies_in_season),
        )
        cdo_command = f"cdo -w -O  ensmean {temp_dir_boot}/* {writedir}"
        os.system(cdo_command)

        # std
        writedir = construct_writedir(
            composite_dir,
            dom,
            filter_lifetime,
            filter_diameter,
            filter_w,
            filter_quantile,
            filter_n,
            stat="boot_std",
            season=season,
            n_cookies=len(cookies_in_season),
        )
        cdo_command = f"cdo -w -O  ensstd {temp_dir_boot}/* {writedir}"
        os.system(cdo_command)

        # # q25
        # writedir = construct_writedir(
        #     composite_dir,
        #     dom,
        #     filter_lifetime,
        #     filter_diameter,
        #     filter_w,
        #     filter_quantile,
        #     filter_n,
        #     stat="boot_q25",
        #     season=season,
        #     n_cookies=len(cookies_in_season),
        # )
        # cdo_command = f"cdo -w -O  enspctl,25 {temp_dir_boot}/* {writedir}"
        # os.system(cdo_command)

        # # q75
        # writedir = construct_writedir(
        #     composite_dir,
        #     dom,
        #     filter_lifetime,
        #     filter_diameter,
        #     filter_w,
        #     filter_quantile,
        #     filter_n,
        #     stat="boot_q75",
        #     season=season,
        #     n_cookies=len(cookies_in_season),
        # )
        # cdo_command = f"cdo -w -O  enspctl,75 {temp_dir_boot}/* {writedir}"
        # os.system(cdo_command)

        # # median
        # writedir = construct_writedir(
        #     composite_dir,
        #     dom,
        #     filter_lifetime,
        #     filter_diameter,
        #     filter_w,
        #     filter_quantile,
        #     filter_n,
        #     stat="boot_median",
        #     season=season,
        #     n_cookies=len(cookies_in_season),
        # )
        # cdo_command = f"cdo -w -O  enspctl,50 {temp_dir_boot}/* {writedir}"
        # os.system(cdo_command)

        # q99
        writedir = construct_writedir(
            composite_dir,
            dom,
            filter_lifetime,
            filter_diameter,
            filter_w,
            filter_quantile,
            filter_n,
            stat="boot_q99",
            season=season,
            n_cookies=len(cookies_in_season),
        )
        cdo_command = f"cdo -w -O  enspctl,99 {temp_dir_boot}/* {writedir}"
        os.system(cdo_command)
        # q01
        writedir = construct_writedir(
            composite_dir,
            dom,
            filter_lifetime,
            filter_diameter,
            filter_w,
            filter_quantile,
            filter_n,
            stat="boot_q01",
            season=season,
            n_cookies=len(cookies_in_season),
        )
        cdo_command = f"cdo -w -O  enspctl,1 {temp_dir_boot}/* {writedir}"
        os.system(cdo_command)

        # remove temp_dir_boot
        for c in os.listdir(temp_dir_boot):
            os.remove(os.path.join(temp_dir_boot, c))
        os.rmdir(temp_dir_boot)
        os.rmdir(temp_dir_resampled)

        # ---------------------------------------------------------
        # calculate straight mean, std, q25, q75, median, q90, q10 from original cookies
        # ---------------------------------------------------------
        # link cookies_in_season to temp_dir
        for c in cookies_in_season:
            os.symlink(c, os.path.join(temp_dir, os.path.basename(c)))

        # mean
        writedir = construct_writedir(
            composite_dir,
            dom,
            filter_lifetime,
            filter_diameter,
            filter_w,
            filter_quantile,
            filter_n,
            stat="mean",
            season=season,
            n_cookies=len(cookies_in_season),
        )
        cdo_command = f"cdo -w -O  ensmean {temp_dir}/* {writedir}"
        # print(cdo_command)
        os.system(cdo_command)
        print(dom, season, len(cookies_in_season))

        # std
        writedir = construct_writedir(
            composite_dir,
            dom,
            filter_lifetime,
            filter_diameter,
            filter_w,
            filter_quantile,
            filter_n,
            stat="std",
            season=season,
            n_cookies=len(cookies_in_season),
        )
        cdo_command = f"cdo -w -O  ensstd {' '.join(cookies_in_season)} {writedir}"
        os.system(cdo_command)

        # q25
        writedir = construct_writedir(
            composite_dir,
            dom,
            filter_lifetime,
            filter_diameter,
            filter_w,
            filter_quantile,
            filter_n,
            stat="q25",
            season=season,
            n_cookies=len(cookies_in_season),
        )
        cdo_command = f"cdo -w -O  enspctl,25 {' '.join(cookies_in_season)} {writedir}"
        os.system(cdo_command)

        # q75
        writedir = construct_writedir(
            composite_dir,
            dom,
            filter_lifetime,
            filter_diameter,
            filter_w,
            filter_quantile,
            filter_n,
            stat="q75",
            season=season,
            n_cookies=len(cookies_in_season),
        )
        cdo_command = f"cdo -w -O  enspctl,75 {' '.join(cookies_in_season)} {writedir}"
        os.system(cdo_command)

        # median
        writedir = construct_writedir(
            composite_dir,
            dom,
            filter_lifetime,
            filter_diameter,
            filter_w,
            filter_quantile,
            filter_n,
            stat="median",
            season=season,
            n_cookies=len(cookies_in_season),
        )
        cdo_command = f"cdo -w -O  enspctl,50 {' '.join(cookies_in_season)} {writedir}"
        os.system(cdo_command)

        # # q90
        # writedir = construct_writedir(
        #     composite_dir,
        #     dom,
        #     filter_lifetime,
        #     filter_diameter,
        #     filter_w,
        #     filter_quantile,
        #     filter_n,
        #     stat="q90",
        #     season=season,
        #     n_cookies=len(cookies_in_season),
        # )
        # cdo_command = f"cdo -w -O  enspctl,90 {' '.join(cookies_in_season)} {writedir}"
        # os.system(cdo_command)

        # remove temp_dir
        for c in os.listdir(temp_dir):
            os.remove(os.path.join(temp_dir, c))
        os.rmdir(temp_dir)

    return


def filter_cookies_cdo(
    cookies_files,
    cookie_dir,
    dom,
    subdomain_dir,
    filter_lifetime=None,
    filter_diameter=None,
    filter_w=None,
    filter_quantile=None,
    filter_n=None,
):
    cookie_directories = [os.path.join(cookie_dir, c) for c in cookies_files]
    # load all cookies
    cookies = xr.open_mfdataset(
        cookie_directories,
        combine="by_coords",
    )
    cookies = filter_cookies(
        cookies,
        dom,
        subdomain_dir,
        filter_lifetime,
        filter_diameter,
        filter_w,
        filter_quantile,
        filter_n,
    )

    cookie_ids = cookies["cookie_id"].values
    cookies_files = [f"cookie_{c}.nc" for c in cookie_ids]

    return cookies_files


def construct_writedir(
    composite_dir,
    dom,
    filter_lifetime=None,
    filter_diameter=None,
    filter_w=None,
    filter_quantile=None,
    filter_n=None,
    stat=None,
    season=None,
    n_cookies=None,
):
    filter_str = ""
    if filter_lifetime is not None:
        filter_str += f"_lifetime{filter_lifetime}"
    if filter_diameter is not None:
        filter_str += f"_max_val{filter_diameter}"
    if filter_w is not None:
        filter_str += f"_w{filter_w}"
    if filter_quantile is not None:
        filter_str += f"_q{filter_quantile}"
    if filter_n is not None:
        filter_str += f"_n{filter_n}"
    if stat is None:
        stat_str = "_comp"
    else:
        stat_str = "_" + stat
    if season is None:
        season_str = ""
    else:
        season_str = "_" + season
    if n_cookies is None:
        n_cookies_str = ""
    else:
        n_cookies_str = "_n" + str(n_cookies)

    writedir = os.path.join(
        composite_dir, dom + season_str + filter_str + stat_str + n_cookies_str + ".nc"
    )
    return writedir


def filter_cookies(
    cookies,
    dom,
    subdomain_dir,
    filter_lifetime=None,
    filter_diameter=None,
    filter_w=None,
    filter_quantile=None,
    filter_n=None,
):
    if filter_lifetime is not None:
        cookies = cookies.where(
            cookies["cell_lifespan"] >= np.timedelta64(filter_lifetime, "m"),
            drop=True,
        )
    if filter_diameter is not None:
        cookies = cookies.where(cookies["max_val"] >= filter_diameter, drop=True)
    if filter_w is not None:
        cookies = cookies.where(
            cookies.sel(pressure=400).W.max(dim=["x", "y"]) >= filter_w, drop=True
        )
    if filter_quantile is not None:
        threshold = np.nanquantile(
            cookies.sel(pressure=400).W.max(dim=["x", "y"]).values, filter_quantile
        )
        cookies = cookies.where(
            cookies.sel(pressure=400).W.max(dim=["x", "y"]) >= threshold, drop=True
        )
    if filter_n is not None:
        # only use the n/1000km^2 per year cookies with the largest hail diameter
        subdomains = xr.open_dataset(
            subdomain_dir,
        )
        dom_area = np.round(np.nansum(subdomains[dom] == 1) * 2.2**2)
        n_cookies = int(dom_area / 1000 * filter_n)

        # select the n cookies with the largest max hail diameter (max of DHAIL_MX variable)
        cookies["sort_val"] = cookies.DHAIL_MX.max(dim=["x", "y"])
        cookies = cookies.sortby("sort_val", ascending=False)
        cookies = cookies.isel(cookie_id=slice(0, n_cookies))

    # only keep cookies after start of cell (filter out genesis cookies)
    cookies = cookies.where(
        cookies.t_rel_start.astype("timedelta64[m]").astype(int) >= 0, drop=True
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
    parser.add_argument('subdomain_dir', type=str, help="Directory containing subdomain file")
    parser.add_argument("composite_dir", type=str, help="Directory to save composites")

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
        "--filter_quantile",
        type=float,
        default=None,
        help="Filter out cookies with a maximum updraft velocity less than this quantile value",
    )
    parser.add_argument(
        "--filter_n",
        type=float,
        default=None,
        help="only use the n/1000km^2 per year cookies with the largest hail diameter",
    )
    parser.add_argument(
        "--serial",
        action="store_true",
        help="Run calculations in serial",
    )
    parser.add_argument(
        "--n_bootstrap",
        type=int,
        default=100,
        help="Number of bootstrap samples to calculate",
    )
    args = parser.parse_args()

    main(
        args.cookie_dir,
        args.subdomain_dir,
        args.composite_dir,
        args.filter_lifetime,
        args.filter_diameter,
        args.filter_w,
        args.filter_quantile,
        args.filter_n,
        args.serial,
        args.n_bootstrap,
    )
