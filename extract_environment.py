#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
---------------------------------------------------------
extracts near cell environment
loops through time instead of cells for efficiency
one file per cell and timestep where it is active and environment is available
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

python /home/kbrennan/cookie_cutter/extract_environment.py /home/kbrennan/phd/data/climate/tracks/present /home/kbrennan/phd/data/climate/present /home/kbrennan/phd/data/climate/cookies/present 20210601 20210630

---------------------------------------------------------
Killian P. Brennan
02.05.2024
version 0.1
---------------------------------------------------------
"""

import os
import json
import xarray as xr
import numpy as np
import pandas as pd
from argparse import ArgumentParser
import metpy.calc as mpcalc
from metpy.units import units


def main(trackdir, envdir, outpath, start_day, end_day, window_radius=25):

    start_day = pd.to_datetime(start_day, format="%Y%m%d")
    end_day = pd.to_datetime(end_day, format="%Y%m%d")

    daylist = pd.date_range(start_day, end_day)

    # make output directory
    os.makedirs(outpath, exist_ok=True)

    # iterate over each day in the dataset
    for day in daylist:
        day_str = day.strftime("%Y%m%d")

        cells = load_cells(trackdir, day_str)

        cells = filter(cells)

        for cell in cells:
            cell["itime"] = 0

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

                    cookie = add_secondary_variables(cookie)

                    cookie = rotate_cookie(cookie, cell)

                    cookie = mask_cookie(cookie, window_radius)

                    cookie = add_attributes(cookie, cell, now)

                    cookie_id = int(
                        (
                            day_str
                            + str(cell["cell_id"]).zfill(5)
                            + str(cell["itime"]).zfill(2)
                        )
                    )
                    cookie = cookie.drop_vars("time")
                    cookie = cookie.expand_dims({"cookie_id": [cookie_id]})
                    write_cookie(cookie, outpath, cookie_id)

                    cell["itime"] += 1
    return


def write_cookie(cookie, outpath, cookie_id):
    comp = dict(zlib=True, complevel=9)

    # change all double variables to float
    for var in cookie.data_vars:
        if cookie[var].dtype == "float64":
            cookie[var] = cookie[var].astype("float32")
            
    encoding = {var: comp for var in cookie.data_vars}
    cookie.to_netcdf(
        os.path.join(outpath, f"cookie_{cookie_id}.nc"),
        encoding=encoding,
    )
    return


def add_secondary_variables(cookie):
    """calculate secondary variables for the cookie
    specific humidity
    dewpoint
    2m dewpoint
    theta
    2m theta
    theta e
    2m theta e
    vorticity
    divergence
    """
    # dewpoint
    cookie["DP"] = xr.DataArray(
        (
            mpcalc.dewpoint_from_relative_humidity(
                cookie["T"] * units.K,
                cookie["RELHUM"] * units.percent,
            )
            / units.K
        ).values,
        dims=["pressure", "y", "x"],
        coords={"y": cookie.y, "x": cookie.x},
    )

    cookie["DP"].attrs = {
        "units": "K",
        "long_name": "dewpoint temperature",
    }

    # 2m dewpoint
    cookie["DP_2M"] = xr.DataArray(
        (
            mpcalc.dewpoint_from_relative_humidity(
                cookie["T_2M"] * units.K,
                cookie["RELHUM_2M"] * units.percent,
            )
            / units.K
        ).values,
        dims=["y", "x"],
        coords={"y": cookie.y, "x": cookie.x},
    )

    cookie["DP_2M"].attrs = {
        "units": "K",
        "long_name": "dewpoint temperature at 2m",
    }

    # theta
    cookie["THETA"] = xr.DataArray(
        (
            mpcalc.potential_temperature(
                cookie.pressure * units.Pa,
                cookie["T"] * units.K,
            )
            / units.K
        ).values,
        dims=["pressure", "y", "x"],
        coords={"y": cookie.y, "x": cookie.x},
    )

    cookie["THETA"].attrs = {
        "units": "K",
        "long_name": "potential temperature",
    }

    # 2m theta
    cookie["THETA_2M"] = xr.DataArray(
        (
            mpcalc.potential_temperature(
                cookie["PS"] * units.Pa,
                cookie["T_2M"] * units.K,
            )
            / units.K
        ).values,
        dims=["y", "x"],
        coords={"y": cookie.y, "x": cookie.x},
    )

    cookie["THETA_2M"].attrs = {
        "units": "K",
        "long_name": "potential temperature at 2m",
    }

    # theta e
    cookie["THETA_E"] = xr.DataArray(
        (
            mpcalc.equivalent_potential_temperature(
                cookie.pressure * units.Pa,
                cookie["T"] * units.K,
                cookie["DP"] * units.K,
            )
            / units.K
        ).values,
        dims=["pressure", "y", "x"],
        coords={"y": cookie.y, "x": cookie.x},
    )

    cookie["THETA_E"].attrs = {
        "units": "K",
        "long_name": "equivalent potential temperature",
    }

    # 2m theta e
    cookie["THETA_E_2M"] = xr.DataArray(
        (
            mpcalc.equivalent_potential_temperature(
                cookie["PS"] * units.Pa,
                cookie["T_2M"] * units.K,
                cookie["DP_2M"] * units.K,
            )
            / units.K
        ).values,
        dims=["y", "x"],
        coords={"y": cookie.y, "x": cookie.x},
    )

    cookie["THETA_E_2M"].attrs = {
        "units": "K",
        "long_name": "equivalent potential temperature at 2m",
    }

    # vorticity
    cookie["VORT"] = xr.DataArray(
        (
            mpcalc.vorticity(
                cookie["U"] * units("m/s"),
                cookie["V"] * units("m/s"),
                dx=2.2 * units.km,
                dy=2.2 * units.km,
            )
            / units("s^-1")
        ).values,
        dims=["pressure", "y", "x"],
    )

    cookie["VORT"].attrs = {
        "units": "s^-1",
        "long_name": "vorticity",
    }

    # divergence
    cookie["DIV"] = xr.DataArray(
        (
            mpcalc.divergence(
                cookie["U"] * units("m/s"),
                cookie["V"] * units("m/s"),
                dx=2.2 * units.km,
                dy=2.2 * units.km,
            )
            / units("s^-1")
        ).values,
        dims=["pressure", "y", "x"],
    )

    cookie["DIV"].attrs = {
        "units": "s^-1",
        "long_name": "divergence",
    }

    return cookie


def add_attributes(cookie, cell, now):
    # add information on the version of cookie creator / extract_environment.py
    cookie.attrs["cookie_creator"] = "extract_environment.py"
    cookie.attrs["cookie_creator_version"] = "0.1"
    cookie.attrs["cookie_creator_version_date"] = "20240502"
    cookie.attrs["cookie_creator_author"] = "Killian P. Brennan"
    # date of creation
    cookie.attrs["creation_date"] = pd.Timestamp.now().strftime("%Y%m%d%H%M")

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
    cookie["real_time"] = now
    cookie["real_time"].assign_attrs(
        {
            "units": "datetime",
            "long_name": "real time of the cell cookie timestep",
        }
    )
    # time in minutes relative to start of the cell
    cookie["t_rel_start"] = (now - cell["datelist"][0]).seconds / 60
    cookie["t_rel_start"].assign_attrs(
        {
            "units": "minutes",
            "long_name": "time in minutes relative to start of the cell",
        }
    )
    # time in minutes relative to end of the cell
    cookie["t_rel_end"] = -(cell["datelist"][-1] - now).seconds / 60
    cookie["t_rel_end"].assign_attrs(
        {
            "units": "minutes",
            "long_name": "time in minutes relative to end of the cell",
        }
    )
    # time relative to maximum intensity of the cell
    t_max = cell["datelist"][np.argmax(cell["max_val"])]
    if t_max < now:
        cookie["t_rel_max"] = (now - t_max).seconds / 60
    else:
        cookie["t_rel_max"] = -(t_max - now).seconds / 60
    cookie["t_rel_max"].assign_attrs(
        {
            "units": "minutes",
            "long_name": "time in minutes relative to maximum intensity of the cell",
        }
    )
    cookie["itime"] = cell["itime"]
    cookie["itime"].assign_attrs(
        {
            "units": "int",
            "long_name": "time index of the cell cookie",
        }
    )

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
        x=-cookie.x * np.sin(angle) + cookie.y * np.cos(angle),
        y=cookie.x * np.cos(angle) + cookie.y * np.sin(angle),
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

    cookie = cookie.squeeze()

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
