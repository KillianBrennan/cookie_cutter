#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
---------------------------------------------------------
extracts near cell environment
loops through time instead of cells for efficiency
---------------------------------------------------------
IN
day to analyze (yyyymmdd)
path to track location
path to environment
path to output
---------------------------------------------------------
OUT
netcdf file with extracted environment (cell centered)
gives one file per cell, named by day_id
---------------------------------------------------------
EXAMPLE CALL
python extract_environment.py 20240412 /path/to/track /path/to/environment /path/to/output

python /home/kbrennan/cookie_cutter/extract_environment.py /home/kbrennan/phd/data/climate/tracks/present /home/kbrennan/phd/data/climate/present /home/kbrennan/phd/data/climate/cookies/present 20210628 20210628

---------------------------------------------------------
Killian P. Brennan
12.04.2024
---------------------------------------------------------
"""

import os
import json
import xarray as xr
import numpy as np
import pandas as pd
from argparse import ArgumentParser


def main(trackdir, envdir, outpath, start_day, end_day, window_radius=25):

    start_day = pd.to_datetime(start_day, format="%Y%m%d")
    end_day = pd.to_datetime(end_day, format="%Y%m%d")

    daylist = pd.date_range(start_day, end_day)

    # make output directory
    os.makedirs(outpath, exist_ok=True)

    # iterate over each day in the dataset
    for day in daylist:
        day_str = day.strftime("%Y%m%d")

        # create temporary directory
        dir_temp = os.path.join(outpath, f"temp_{day_str}")
        os.makedirs(dir_temp, exist_ok=True)

        cells = load_cells(trackdir, day_str)

        cells = filter(cells)

        times = [day + pd.Timedelta(hours=i) for i in range(24)]

        for now in times:
            print(now)
            loaded = False
            for cell in cells:
                if now in cell["datelist"]:
                    if not loaded:
                        env = load_environments(envdir, now)
                        loaded = True

                    cookie = cutout_cookie(cell, env, now, window_radius=window_radius)

                    # skip if cookie is None (bbox outside of domain)
                    if cookie is None:
                        continue

                    cookie = rotate_cookie(cookie, cell)

                    cookie = mask_cookie(cookie, window_radius)

                    # save environment
                    unique_id = int((day_str + str(cell["cell_id"]).zfill(5)))
                    cookie.expand_dims({"unique_id": [unique_id]})

                    # make dir for cell
                    os.makedirs(
                        os.path.join(dir_temp, f"{unique_id}"),
                        exist_ok=True,
                    )

                    cookie.to_netcdf(
                        os.path.join(
                            dir_temp,
                            str(unique_id),
                            f"{day_str}{now.hour}.nc",
                        ),
                    )

        # merge all files for each cell
        dirs = os.listdir(dir_temp)
        for id in dirs:
            # delete file if it already exists
            if os.path.exists(os.path.join(outpath, f"cookie_{id}.nc")):
                os.remove(os.path.join(outpath, f"cookie_{id}.nc"))
            # os.system(
            #     "cdo mergetime "
            #     + os.path.join(dir_temp, id, "*.nc")
            #     + " "
            #     + os.path.join(outpath, f"cookie_{id}.nc")
            # )
            cookie = merge_files(id, dir_temp)
            cell = [cell for cell in cells if cell["cell_id"] == int(id[-4:])][0]
            cookie = add_attributes(cookie, cell)

            # cookie = make_itime(cookie)
            n_timesteps = len(cookie.time)
            # compute mean along time dimension
            cookie = cookie.mean(dim="time", keep_attrs=True, skipna=True)
            # add cell_id dimension
            cookie = cookie.expand_dims({"cell_id": [id]})

            # add attribute on how many timesteps are included
            cookie["n_timesteps"] = n_timesteps
            # add more details on n_timesteps
            cookie["n_timesteps"].assign_attrs(
                {
                    "units": "hours",
                    "long_name": "number of hourly timesteps included in the cookie",
                }
            )

            comp = dict(zlib=True, complevel=9)
            encoding = {var: comp for var in cookie.data_vars}
            cookie.to_netcdf(
                os.path.join(outpath, f"cookie_{id}.nc"), encoding=encoding
            )

        # remove temporary directory
        os.system(f"rm -r {dir_temp}")

    return


def merge_files(id, dir_temp):
    files = os.listdir(os.path.join(dir_temp, id))
    cookie = xr.open_mfdataset(
        [os.path.join(dir_temp, id, file) for file in files],
        combine="nested",
        concat_dim="time",
    )
    return cookie


def make_itime(cookie):
    """
    make integer time index
    and replace time with itime
    """
    cookie = cookie.rename({"time": "itime"})
    cookie.coords["itime"] = np.arange(cookie.itime.size)

    return cookie


def add_attributes(cookie, cell):
    # add lat / lon of maximum intensity
    cookie["lon_max"] = cell["lon"][np.argmax(cell["max_val"])]
    cookie["lat_max"] = cell["lat"][np.argmax(cell["max_val"])]
    # add information on lat_max and lon_max
    cookie["lat_max"].assign_attrs(
        {
            "units": "degrees",
            "long_name": "latitude of maximum intensity",
        }
    )
    cookie["lon_max"].assign_attrs(
        {
            "units": "degrees",
            "long_name": "longitude of maximum intensity",
        }
    )

    # add cell attributes
    cookie["cell_lifespan"] = cell["lifespan"]
    cookie["cell_lifespan"].assign_attrs(
        {
            "units": "minutes",
            "long_name": "lifespan of the cell",
        }
    )

    cookie["max_val"] = np.max(cell["max_val"])
    cookie["max_val"].assign_attrs(
        {
            "units": "mm",
            "long_name": "maximum hail diameter produced by the cell",
        }
    )
    # cookie["cell_start_time"] = cell["datelist"][0].strftime("%Y-%m-%d %H:%M:%S")
    # cookie["cell_start_time"].assign_attrs(
    #     {
    #         "long_name": "start time of the cell",
    #     }
    # )
    return cookie


def filter(cells, min_lifespan=60):
    # filter cells by lifespan
    cells = [cell for cell in cells if cell["lifespan"] > min_lifespan]

    return cells


def mask_cookie(cookie, radius_gp):
    radius_km = radius_gp * 2.2
    masked = cookie.where(np.sqrt(cookie.x**2 + cookie.y**2) <= radius_km)
    return masked


def rotate_cookie(cookie, cell):
    "rotate cookie to have the mean cell propagation vector in the x direction"

    # get the cell propagation vector mean over lifetime, first has no movement vector
    u_storm = np.nanmean(cell["delta_y"][1::])
    v_storm = np.nanmean(cell["delta_x"][1::])
    # calculate the angle
    angle = np.arctan2(v_storm, u_storm)
    # rotate the cookie
    cookie = cookie.interp(
        x=cookie.x * np.cos(angle) + cookie.y * np.sin(angle),
        y=-cookie.x * np.sin(angle) + cookie.y * np.cos(angle),
    )

    # u and v wind components need to be rotated as well
    u = cookie["U"]
    v = cookie["V"]
    cookie["U"] = u * np.cos(angle) + v * np.sin(angle)
    cookie["V"] = -u * np.sin(angle) + v * np.cos(angle)

    # 10m wind components need to be rotated as well
    u = cookie["U_10M"]
    v = cookie["V_10M"]
    cookie["U_10M"] = u * np.cos(angle) + v * np.sin(angle)
    cookie["V_10M"] = -u * np.sin(angle) + v * np.cos(angle)

    return cookie


def cutout_cookie(cell, env, now, window_radius=25):
    # find the index where now is
    i = cell["datelist"].index(now)

    x = np.round(cell["mass_center_y"][i])
    y = np.round(cell["mass_center_x"][i])
    bbox = [
        x - window_radius,
        x + window_radius,
        y - window_radius,
        y + window_radius,
    ]
    bbox = [int(b) for b in bbox]

    # check wether the bbox is within the environment domain and return None if not
    if (
        bbox[0] < 0
        or bbox[2] < 0
        or bbox[1] >= env.rlon.size
        or bbox[3] >= env.rlat.size
    ):
        return None

    # extract cookie
    cookie = env.isel(
        rlat=slice(bbox[2], bbox[3] + 1),
        rlon=slice(bbox[0], bbox[1] + 1),
    )

    # rename coordinates
    cookie = cookie.rename({"rlat": "y", "rlon": "x"})

    # change coordinates to be cell centered
    cookie["x"] = np.arange(-window_radius, window_radius + 1) * 2.2
    cookie["y"] = np.arange(-window_radius, window_radius + 1) * 2.2

    cookie = cookie.drop_vars(["lat", "lon"])
    cookie = cookie.drop_dims(["bnds", "level1", "windsector", "soil1"])

    return cookie


def load_environments(envdir, now):
    env_1h_2D = xr.open_dataset(
        os.path.join(envdir, "1h_2D", f"lffd{now.strftime('%Y%m%d%H%M')}00.nc")
    )
    env_1h_2D_LPI = xr.open_dataset(
        os.path.join(envdir, "1h_2D_LPI", f"lffd{now.strftime('%Y%m%d%H%M')}00.nc")
    )

    env_1h_3D_plev = xr.open_mfdataset(
        os.path.join(envdir, "1h_3D_plev", f"lffd{now.strftime('%Y%m%d%H%M')}00p.nc"),
        combine="by_coords",
    )
    env_5min_2D = xr.open_mfdataset(
        os.path.join(envdir, "5min_2D", f"lffd{now.strftime('%Y%m%d%H%M')}00.nc"),
        combine="by_coords",
    )
    # drop 'height_toa', 'height_10m', 'height_2m' from 1h_2D (they are only 1D)
    env_1h_2D = env_1h_2D.drop_vars(["height_toa", "height_10m", "height_2m"])

    env = xr.merge(
        [env_1h_2D, env_1h_2D_LPI, env_1h_3D_plev, env_5min_2D],
        combine_attrs="override",
        compat="override",
    )

    return env


def load_cells(trackdir, day):
    with open(os.path.join(trackdir, f"cell_tracks_{day}.json"), "r") as f:
        track_data = json.load(f)

    cells = track_data["cell_data"]

    for cell in cells:
        cell["datelist"] = list(pd.to_datetime(cell["datelist"]))

    return cells


if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument("track", type=str, help="path to track location")
    p.add_argument("env", type=str, help="path to environment")
    p.add_argument("out", type=str, help="path to output")
    p.add_argument("start_day", type=str, help="start day, format: YYYYMMDD")
    p.add_argument("end_day", type=str, help="end day, format: YYYYMMDD")
    p.add_argument(
        "--window_radius",
        type=int,
        default=25,
        help="radius of the window to extract environment",
    )

    P = p.parse_args()
    main(P.track, P.env, P.out, P.start_day, P.end_day, P.window_radius)
