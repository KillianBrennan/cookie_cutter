#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
---------------------------------------------------------
calculate composites from cookies
uses boostrapping to determine significance of difference between present and future composite means
---------------------------------------------------------
IN

---------------------------------------------------------
OUT

---------------------------------------------------------
EXAMPLE CALL
python /home/kbrennan/cookie_cutter/calculate_composites.py /home/kbrennan/phd/data/climate/cookies/ /home/kbrennan/phd/data/climate/grids/subdomains_lonlat.nc /home/kbrennan/phd/data/climate/composites/comp_p0.9 --filter_quantile 0.9
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
    n_bootstrap=1000,
    parallel_bootstrap=True,
):
    print("n_bootstrap", n_bootstrap)
    os.makedirs(composite_dir, exist_ok=True)

    cookies_dir_present = os.path.join(cookie_dir, "present", "subdomains")
    cookies_dir_future = os.path.join(cookie_dir, "future", "subdomains")

    if not os.path.exists(cookies_dir_present):
        print(
            f"subdomains directory {cookies_dir_present} does not exist, did you run assign_subdomains.py?"
        )
        sys.exit()

    if not os.path.exists(cookies_dir_future):
        print(
            f"subdomains directory {cookies_dir_future} does not exist, did you run assign_subdomains.py?"
        )
        sys.exit()

    subdomains_future = os.listdir(cookies_dir_future)
    # must be directories
    subdomains_future = [
        sub
        for sub in subdomains_future
        if os.path.isdir(os.path.join(cookies_dir_future, sub))
    ]

    subdomains_present = os.listdir(cookies_dir_present)
    # must be directories
    subdomains_present = [
        sub
        for sub in subdomains_present
        if os.path.isdir(os.path.join(cookies_dir_present, sub))
    ]

    # only use subdomains that are in both present and future
    subdomains = list(set(subdomains_present) & set(subdomains_future))
    # print warning if some subdomains are missing
    if len(subdomains) != len(subdomains_present):
        print(
            "Warning: some subdomains are missing in the future directory. Only using subdomains that are present in both present and future directories."
        )

    # sort alphabetically
    subdomains.sort()
    # calculate ALL first
    if "ALL" in subdomains:
        subdomains.remove("ALL")
        subdomains = ["ALL"] + subdomains

    # subdomains = ["BI",'NA'] # for testing

    print("subdomains", subdomains)

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
            parallel_bootstrap,
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
    n_bootstrap=1000,
    parallel_bootstrap=True,
):

    # seasons = ["DJF", "MAM", "JJA", "SON", "YEAR"]
    # seasons = ["DJF", "MAM", "JJA", "SON"]
    # seasons = ["MAM", "JJA", "SON"]
    # seasons = ['YEAR']
    seasons = ["JJA"]
    months = {
        "DJF": "12,01,02",
        "MAM": "03,04,05",
        "JJA": "06,07,08",
        "SON": "09,10,11",
        "YEAR": "01,02,03,04,05,06,07,08,09,10,11,12",
    }

    cookie_dir_present = os.path.join(cookie_dir, "present")
    cookie_dir_future = os.path.join(cookie_dir, "future")

    all_cookies_present = os.listdir(
        os.path.join(cookie_dir_present, "subdomains", dom)
    )
    all_cookies_present = filter_cookies_cdo(
        all_cookies_present,
        cookie_dir_present,
        dom,
        subdomain_dir,
        filter_lifetime,
        filter_diameter,
        filter_w,
        filter_quantile,
        filter_n,
    )

    all_cookies_future = os.listdir(os.path.join(cookie_dir_future, "subdomains", dom))
    all_cookies_future = filter_cookies_cdo(
        all_cookies_future,
        cookie_dir_future,
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
        cookies_in_season_present = [
            c for c in all_cookies_present if c[11:13] in months[season]
        ]
        cookies_in_season_present = [
            os.path.join(cookie_dir_present, "subdomains", dom, c)
            for c in cookies_in_season_present
        ]

        cookies_in_season_future = [
            c for c in all_cookies_future if c[11:13] in months[season]
        ]
        cookies_in_season_future = [
            os.path.join(cookie_dir_future, "subdomains", dom, c)
            for c in cookies_in_season_future
        ]

        n_cookies_present = len(cookies_in_season_present)
        n_cookies_future = len(cookies_in_season_future)

        # make temporary directory
        temp_dir_present = os.path.join(
            composite_dir, "temp_present_" + dom + "_" + season
        )
        os.makedirs(temp_dir_present, exist_ok=True)

        temp_dir_future = os.path.join(
            composite_dir, "temp_future_" + dom + "_" + season
        )
        os.makedirs(temp_dir_future, exist_ok=True)

        # ---------------------------------------------------------
        # calculate bootstrap composites
        # ---------------------------------------------------------
        # resample 1000 times with replacement and calculate the mean from the resampled cookies
        # from those means calculate the difference between present and future (for each pair of resampled means)
        # calculate q95 to q05 of the difference
        # means are significantly different (2sigma) if the q05 to q95 interval does not contain 0

        print("n_bootstrap", n_bootstrap)

        # make temporary directory for the bootstrapped means
        temp_dir_boot_present = os.path.join(temp_dir_present, "boot")
        os.makedirs(temp_dir_boot_present, exist_ok=True)

        temp_dir_boot_future = os.path.join(temp_dir_future, "boot")
        os.makedirs(temp_dir_boot_future, exist_ok=True)

        # make temporary directory for the difference
        temp_dir_boot_diff = os.path.join(
            composite_dir, "temp_diff_" + dom + "_" + season
        )
        os.makedirs(temp_dir_boot_diff, exist_ok=True)

        if parallel_bootstrap:
            print("calculating bootstrap means")
            # calculate bootstrapped means
            with mp.Pool(mp.cpu_count() // 3 * 2) as pool:
                pool.starmap(
                    bootstrap,
                    [
                        (
                            temp_dir_boot_present,
                            temp_dir_present,
                            cookies_in_season_present,
                            i,
                        )
                        for i in range(n_bootstrap)
                    ],
                )
            with mp.Pool(mp.cpu_count() // 3 * 2) as pool:
                pool.starmap(
                    bootstrap,
                    [
                        (
                            temp_dir_boot_future,
                            temp_dir_future,
                            cookies_in_season_future,
                            i,
                        )
                        for i in range(n_bootstrap)
                    ],
                )
            # calculate difference between present and future
            print("calculating bootstrap differences")
            with mp.Pool(mp.cpu_count() // 3 * 2) as pool:
                pool.starmap(
                    calculate_bootstrap_diff,
                    [
                        (
                            temp_dir_boot_future,
                            temp_dir_boot_present,
                            temp_dir_boot_diff,
                            i,
                        )
                        for i in range(n_bootstrap)
                    ],
                )
        else:
            # calculate bootstrapped means
            for i in range(n_bootstrap):
                bootstrap(
                    temp_dir_boot_present,
                    temp_dir_present,
                    cookies_in_season_present,
                    i,
                )
            for i in range(n_bootstrap):
                bootstrap(
                    temp_dir_boot_future, temp_dir_future, cookies_in_season_future, i
                )

            # calculate difference between present and future
            for i in range(n_bootstrap):
                calculate_bootstrap_diff(
                    temp_dir_boot_future, temp_dir_boot_present, temp_dir_boot_diff, i
                )

        # bootstrap mean
        cdo_command = f"cdo -O  ensmean {temp_dir_boot_present}/* {composite_dir}/present_mean_{dom}_{season}_n{n_cookies_present}.nc"
        os.system(cdo_command)
        cdo_command = f"cdo -O  ensmean {temp_dir_boot_future}/* {composite_dir}/future_mean_{dom}_{season}_n{n_cookies_future}.nc"
        os.system(cdo_command)

        # calculate q95 and q05 of the difference
        cdo_command = f"cdo -O  enspctl,95 {temp_dir_boot_diff}/* {composite_dir}/diff_q95_{dom}_{season}_n{n_bootstrap}.nc"
        os.system(cdo_command)
        cdo_command = f"cdo -O  enspctl,5 {temp_dir_boot_diff}/* {composite_dir}/diff_q05_{dom}_{season}_n{n_bootstrap}.nc"
        os.system(cdo_command)

        # # bootstrap std
        # cdo_command = f"cdo -O  ensstd {temp_dir_boot_present}/* {composite_dir}/present_std_{dom}_{season}_n{n_cookies_present}.nc"
        # os.system(cdo_command)
        # cdo_command = f"cdo -O  ensstd {temp_dir_boot_future}/* {composite_dir}/future_std_{dom}_{season}_n{n_cookies_future}.nc"
        # os.system(cdo_command)

        # # q95
        # cdo_command = f"cdo -O  enspctl,95 {temp_dir_boot_present}/* {composite_dir}/present_q95_{dom}_{season}_n{n_cookies_present}.nc"
        # os.system(cdo_command)
        # cdo_command = f"cdo -O  enspctl,95 {temp_dir_boot_future}/* {composite_dir}/future_q95_{dom}_{season}_n{n_cookies_future}.nc"
        # os.system(cdo_command)
        # # q05
        # cdo_command = f"cdo -O  enspctl,5 {temp_dir_boot_present}/* {composite_dir}/present_q05_{dom}_{season}_n{n_cookies_present}.nc"
        # os.system(cdo_command)
        # cdo_command = f"cdo -O  enspctl,5 {temp_dir_boot_future}/* {composite_dir}/future_q05_{dom}_{season}_n{n_cookies_future}.nc"

        # remove temp_dir_boot
        for c in os.listdir(temp_dir_boot_present):
            os.remove(os.path.join(temp_dir_boot_present, c))
        os.rmdir(temp_dir_boot_present)
        for c in os.listdir(temp_dir_boot_future):
            os.remove(os.path.join(temp_dir_boot_future, c))
        os.rmdir(temp_dir_boot_future)

        # remove temp_dir_boot_diff
        for c in os.listdir(temp_dir_boot_diff):
            os.remove(os.path.join(temp_dir_boot_diff, c))
        os.rmdir(temp_dir_boot_diff)

        # remove temp_dir_present
        for c in os.listdir(temp_dir_present):
            os.remove(os.path.join(temp_dir_present, c))
        os.rmdir(temp_dir_present)

        # remove temp_dir_future
        for c in os.listdir(temp_dir_future):
            os.remove(os.path.join(temp_dir_future, c))
        os.rmdir(temp_dir_future)
    return


def calculate_bootstrap_diff(
    temp_dir_boot_future, temp_dir_boot_present, temp_dir_boot_diff, i
):
    niceness = os.nice(0)
    os.nice(10 - niceness)

    cdo_command = f"cdo -O -w sub {temp_dir_boot_future}/boot_{i}.nc {temp_dir_boot_present}/boot_{i}.nc {temp_dir_boot_diff}/diff_{i}.nc"
    os.system(cdo_command)
    return


def bootstrap(temp_dir_boot, temp_dir, cookies_in_season, i):

    niceness = os.nice(0)
    os.nice(10 - niceness)

    # do bootstrapping for one iteration

    # make temporary directory for the resampled cookies
    temp_dir_resampled = os.path.join(temp_dir, "resampled_" + str(i))
    os.makedirs(temp_dir_resampled, exist_ok=True)

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
        f"cdo -O -w ensmean  {temp_dir_resampled}/* {temp_dir_boot}/boot_{i}.nc"
    )
    os.system(cdo_command)

    # remove resampled cookies
    for c in os.listdir(temp_dir_resampled):
        os.remove(os.path.join(temp_dir_resampled, c))
    os.rmdir(temp_dir_resampled)
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
    print("filtering cookies")

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
    # only keep cookies after start of cell (filter out genesis cookies) this needs to be done before any other filtering (quantile would be wrong otherwise)
    cookies = cookies.where(
        cookies.t_rel_start.astype("timedelta64[m]").astype(int) >= 0, drop=True
    )
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
        # threshold = np.nanquantile(
        #     cookies.sel(pressure=400).W.max(dim=["x", "y"]).values, filter_quantile
        # )
        # cookies = cookies.where(
        #     cookies.sel(pressure=400).W.max(dim=["x", "y"]) >= threshold, drop=True
        # )
        threshold = np.nanquantile(
            cookies.DHAIL_MX.max(dim=["x", "y"]).values, filter_quantile
        )
        cookies = cookies.where(
            cookies.DHAIL_MX.max(dim=["x", "y"]) >= threshold, drop=True
        )
    if filter_n is not None:
        # only use the n/1000km^2 per year cookies with the largest hail diameter
        # print('subdomain_dir', subdomain_dir)
        subdomains = xr.open_dataset(
            subdomain_dir,
        )
        dom_area = np.round(np.nansum(subdomains[dom] == 1) * 2.2**2)
        n_cookies = int(dom_area / 1000 * filter_n)
        print("n_cookies", n_cookies)

        # select the n cookies with the largest max hail diameter (max of DHAIL_MX variable)
        cookies["sort_val"] = cookies.DHAIL_MX.max(dim=["x", "y"])
        cookies = cookies.sortby("sort_val", ascending=False)
        cookies = cookies.isel(cookie_id=slice(0, n_cookies))
        print("len(cookies.cookie_id)", len(cookies.cookie_id))

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate composites from cookies")
    parser.add_argument("cookie_dir", type=str, help="Directory containing cookies")
    parser.add_argument(
        "subdomain_dir", type=str, help="Directory containing subdomain file"
    )
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
        "--n_bootstrap",
        type=int,
        default=1000,
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
        args.n_bootstrap,
    )
