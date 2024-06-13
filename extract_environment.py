#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
---------------------------------------------------------
extracts near cell environment
loops through time instead of cells for efficiency
one file per cell and timestep where it is active and environment is available
filters only cells with lifespan > 60 minutes
adds 30 minutes of genesis time to each cell (to aid in genesis analysis)
rotates the cookie to have the mean cell propagation vector in the y direction
mask disk around the cell center
mask topography (values below surface pressure are masked)
adds secondary variables (dewpoint, specific humidity, theta, theta e, vorticity, divergence)
currently only cuts out 1hourly fields, 5min fields are used but only hourly
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

python /home/kbrennan/cookie_cutter/extract_environment.py /home/kbrennan/phd/data/climate/tracks/present /home/kbrennan/phd/data/climate/present /home/kbrennan/phd/data/climate/cookies/present/test 20210601 20210630
---------------------------------------------------------
KNOWN ISSUES
will throw some dask runtime warnings, these can be ignored
---------------------------------------------------------
Killian P. Brennan
13.06.2024
version 0.2
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


def main(trackdir, envdir, outpath, start_day, end_day, window_radius=30):
    """
    main function to extract the environment

    Parameters
    ----------
    trackdir : str
    path to the track location
    envdir : str
    path to the environment
    outpath : str
    path to output
    start_day : str
    start day, format: YYYYMMDD
    end_day : str
    end day, format: YYYYMMDD
    window_radius : int
    radius of the window to extract environment (gridpoints)
    """

    start_day = pd.to_datetime(start_day, format="%Y%m%d")
    end_day = pd.to_datetime(end_day, format="%Y%m%d")

    daylist = pd.date_range(start_day, end_day)

    # make output directory
    os.makedirs(outpath, exist_ok=True)

    # load constants file
    const = load_constants(envdir)

    # iterate over each day in the dataset
    for day in daylist:
        day_str = day.strftime("%Y%m%d")

        cells = load_cells(trackdir, day_str)

        cells = filter(cells)

        cells = [add_genesis(cell) for cell in cells]

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
                        env = xr.merge(
                            [env, const], combine_attrs="override", compat="override"
                        )
                        loaded = True

                    cookie = cutout_cookie(cell, env, now, window_radius=window_radius)

                    # skip if cookie is None (bbox outside of domain)
                    if cookie is None:
                        continue

                    cookie = add_secondary_variables(cookie)

                    cookie = rotate_cookie(cookie, cell)

                    cookie = mask_disk(cookie, window_radius)

                    cookie = mask_topography(cookie)

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
                    cookie["cookie_id"].attrs = {
                        "long_name": "unique identifier for the cookie, constructed from YYYYMMDD + cell_id.zfill(5) + itime.zfill(2)"
                    }

                    write_cookie(cookie, outpath, cookie_id)

                    cell["itime"] += 1
    return


def load_constants(envdir):
    """
    loads the constants file
    """
    # load first file in directory that ends with *c.nc
    for file in os.listdir(envdir):
        if file.endswith("c.nc"):
            const = xr.open_dataset(os.path.join(envdir, file))
            break

    # only keep variables in list
    varlist = ["S_ORO_MAX", "HSURF"]
    const = const[varlist]
    # remove time dimension
    const = const.squeeze()
    const = const.drop_vars("time")
    return const


def mask_topography(cookie):
    """
    masks values that lie below the topography (based on pressure level being higher than surface pressure)
    """
    # mask 3d variables that are below the surface

    for key in cookie.keys():
        # if the variable is 3d
        if "pressure" in cookie[key].coords:
            # mask the variable
            cookie[key] = xr.where(
                cookie.pressure < cookie.PS, cookie[key], np.nan, keep_attrs=True
            )

    return cookie


def write_cookie(cookie, outpath, cookie_id):
    """
    writes the cookie to a netcdf file
    """
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
    dewpoint
    2m dewpoint
    specific humidity
    2m specific humidity
    theta
    2m theta
    theta e
    2m theta e
    vorticity
    divergence
    """

    # dewpoint
    cookie["TD"] = xr.DataArray(
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

    cookie["TD"].attrs = {
        "units": "K",
        "long_name": "dewpoint temperature",
    }

    # specific humidity
    cookie["QV"] = xr.DataArray(
        (
            mpcalc.specific_humidity_from_dewpoint(
                cookie.pressure * units.hPa,
                cookie["TD"] * units.K,
            )
            / units("kg/kg")
        ).values,
        dims=["pressure", "y", "x"],
        coords={"y": cookie.y, "x": cookie.x},
    )

    cookie["QV"].attrs = {
        "units": "kg/kg",
        "long_name": "specific humidity",
    }

    # 2m specific humidity
    cookie["QV_2M"] = xr.DataArray(
        (
            mpcalc.specific_humidity_from_dewpoint(
                cookie["PS"] * units.hPa,
                cookie["TD_2M"] * units.K,
            )
            / units("kg/kg")
        ).values,
        dims=["y", "x"],
        coords={"y": cookie.y, "x": cookie.x},
    )

    cookie["QV_2M"].attrs = {
        "units": "kg/kg",
        "long_name": "specific humidity at 2m",
    }

    # theta
    cookie["THETA"] = xr.DataArray(
        (
            mpcalc.potential_temperature(
                cookie.pressure * units.hPa,
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
                cookie["PS"] * units.hPa,
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
                cookie.pressure * units.hPa,
                cookie["T"] * units.K,
                cookie["TD"] * units.K,
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
                cookie["PS"] * units.hPa,
                cookie["T_2M"] * units.K,
                cookie["TD_2M"] * units.K,
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
    """
    adds attributes to the cookie
    like information on the version of cookie creator / extract_environment.py
    date of creation
    information on the cell-centered coordinate system
    and some cell event relative times
    """
    # add information on the version of cookie creator / extract_environment.py
    cookie.attrs["cookie_creator"] = "extract_environment.py"
    cookie.attrs["cookie_creator_version"] = "0.1"
    cookie.attrs["cookie_creator_version_date"] = "20240502"
    cookie.attrs["cookie_creator_author"] = "Killian P. Brennan"
    # date of creation
    cookie.attrs["creation_date"] = pd.Timestamp.now().strftime("%Y%m%d%H%M")
    # information on the cell-centered coordinate system
    cookie.attrs["horizontal coordinate system"] = (
        "x and y are relative to the cell center, y is aligned with storm motion, x is perpendicular to storm motion, positive y is in storm direction, positive x is to the left of the storm motion."
    )
    cookie.attrs["vertical coordinate system"] = "pressure levels are in hPa"

    # add lat / lon of maximum intensity
    cookie["lon_max"] = cell["lon"][np.argmax(cell["max_val"])]
    cookie["lat_max"] = cell["lat"][np.argmax(cell["max_val"])]
    # add information on lat_max and lon_max
    cookie["lat_max"].attrs = {
        "units": "degrees_north",
        "long_name": "latitude of maximum intensity",
    }
    cookie["lon_max"].attrs = {
        "units": "degrees_east",
        "long_name": "longitude of maximum intensity",
    }

    # add cell attributes
    cookie["cell_lifespan"] = cell["lifespan"]
    cookie["cell_lifespan"].attrs = {
        "units": "minutes",
        "long_name": "lifespan of the cell",
    }

    cookie["max_val"] = np.max(cell["max_val"])
    cookie["max_val"].attrs = {
        "units": "mm",
        "long_name": "maximum intensity of the cell",
    }
    cookie["real_time"] = now
    # cookie["real_time"].attrs = {
    #     "units": "datetime",
    #     "long_name": "real time of the cookie timestep",
    # }
    # time in minutes relative to start of the cell
    cookie["t_rel_start"] = (now - cell["start_date"]).seconds / 60
    cookie["t_rel_start"].attrs = {
        "units": "minutes",
        "long_name": "time in minutes relative to start of the cell",
    }

    # time in minutes relative to end of the cell
    cookie["t_rel_end"] = -(cell["datelist"][-1] - now).seconds / 60
    cookie["t_rel_end"].attrs = {
        "units": "minutes",
        "long_name": "time in minutes relative to end of the cell",
    }
    # time relative to maximum intensity of the cell
    t_max = cell["datelist"][np.argmax(cell["max_val"])]
    if t_max < now:
        cookie["t_rel_max"] = (now - t_max).seconds / 60
    else:
        cookie["t_rel_max"] = -(t_max - now).seconds / 60
    cookie["t_rel_max"].attrs = {
        "units": "minutes",
        "long_name": "time in minutes relative to maximum intensity of the cell",
    }

    cookie["itime"] = cell["itime"]
    cookie["itime"].attrs = {
        "units": "index",
        "long_name": "time index of the cell cookie",
    }

    return cookie


def filter(cells, min_lifespan=60):
    """
    filters cells by lifespan
    """
    # filter cells by lifespan
    cells = [cell for cell in cells if cell["lifespan"] > min_lifespan]

    return cells


def mask_disk(cookie, radius_gp):
    """
    masks a disk around the cell center
    """
    radius_km = radius_gp * 2.2
    masked = cookie.where(np.sqrt(cookie.x**2 + cookie.y**2) <= radius_km)
    return masked


def rotate_cookie(cookie, cell):
    """
    rotates the cookie to have the mean cell propagation vector in the y direction
    will also rotate the wind components
    """

    # get the cell propagation vector mean over lifetime, first has no movement vector (x and y components are flipped from the tracker...)
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

    cookie["U"].attrs = {
        "units": "m/s",
        "long_name": "wind component perpendicular to storm motion, positive is to the left of the storm motion",
    }
    cookie["V"].attrs = {
        "units": "m/s",
        "long_name": "wind component aligned with storm motion, positive is in storm direction",
    }

    # 10m wind components need to be rotated as well
    u = cookie["U_10M"]
    v = cookie["V_10M"]
    cookie["U_10M"] = u * np.cos(angle) + v * np.sin(angle)
    cookie["V_10M"] = -u * np.sin(angle) + v * np.cos(angle)

    cookie["U_10M"].attrs = {
        "units": "m/s",
        "long_name": "10m wind component perpendicular to storm motion, positive is to the left of the storm motion",
    }
    cookie["V_10M"].attrs = {
        "units": "m/s",
        "long_name": "10m wind component aligned with storm motion, positive is in storm direction",
    }

    return cookie


def cutout_cookie(cell, env, now, window_radius=25):
    """
    cuts out the cookie around the cell center
    """
    # find the index where now is
    i = cell["datelist"].index(now)

    x = np.round(cell["mass_center_y"][i])
    y = np.round(cell["mass_center_x"][i])
    bbox = [
        x - window_radius,
        x + window_radius + 1,
        y - window_radius,
        y + window_radius + 1,
    ]
    bbox = [int(b) for b in bbox]

    # extract cookie
    cookie = domain_exiting_isel(env, bbox)

    rlat = cookie.rlat.values
    rlon = cookie.rlon.values
    lat = cookie.lat.values
    lon = cookie.lon.values

    # rename coordinates
    cookie = cookie.rename({"rlat": "y", "rlon": "x"})

    # change coordinates to be cell centered
    cookie["x"] = np.arange(-window_radius, window_radius + 1) * 2.2
    cookie["y"] = np.arange(-window_radius, window_radius + 1) * 2.2

    cookie["x"].attrs = {
        "units": "km",
        "long_name": "x coordinate relative to cell center, aligned with storm motion, positive x is to the left of the storm motion",
        "left": "positive",
    }
    cookie["y"].attrs = {
        "units": "km",
        "long_name": "y coordinate relative to cell center, perpendicular to storm motion, positive y is in storm direction",
        "forwards": "positive",
    }

    cookie["rlat"] = xr.DataArray(rlat, dims="y")
    cookie["rlon"] = xr.DataArray(rlon, dims="x")

    cookie["rlat"].attrs = {
        "units": "degrees_north",
        "long_name": "latitude in rotated pole grid",
    }
    cookie["rlon"].attrs = {
        "units": "degrees_east",
        "long_name": "longitude in rotated pole grid",
    }

    # remove lat and lon
    cookie = cookie.drop_vars(["lat", "lon"])

    cookie["lat"] = xr.DataArray(lat, dims=["y", "x"])
    cookie["lon"] = xr.DataArray(lon, dims=["y", "x"])

    cookie["lat"].attrs = {"units": "degrees_north", "long_name": "latitude"}
    cookie["lon"].attrs = {"units": "degrees_east", "long_name": "longitude"}

    return cookie


def domain_exiting_isel(env, bbox):
    """
    like xr.isel but the bbox can be outside of the domain
    """
    box_size = bbox[1] - bbox[0]
    bbox_save = bbox.copy()
    bbox_save[0] = max(0, bbox_save[0])
    bbox_save[1] = min(env.rlon.size, bbox_save[1])
    bbox_save[2] = max(0, bbox_save[2])
    bbox_save[3] = min(env.rlat.size, bbox_save[3])

    cookie = env.isel(
        rlat=slice(bbox_save[2], bbox_save[3]),
        rlon=slice(bbox_save[0], bbox_save[1]),
    )

    # add nans for the part of the cookie that is outside of the domain to padd it to box_size

    if bbox[0] < 0:
        print("exiting domain")
        for i in range(bbox_save[0] - bbox[0]):
            cookie = xr.concat(
                [xr.full_like(cookie.isel(rlon=0), np.nan), cookie], dim="rlon"
            )
    if bbox[1] >= env.rlon.size:
        print("exiting domain")
        for i in range(bbox[1] - bbox_save[1]):
            cookie = xr.concat(
                [cookie, xr.full_like(cookie.isel(rlon=-1), np.nan)], dim="rlon"
            )
    if bbox[2] < 0:
        print("exiting domain")
        for i in range(bbox_save[2] - bbox[2]):
            cookie = xr.concat(
                [xr.full_like(cookie.isel(rlat=0), np.nan), cookie], dim="rlat"
            )
    if bbox[3] >= env.rlat.size:
        print("exiting domain")
        for i in range(bbox[3] - bbox_save[3]):
            cookie = xr.concat(
                [cookie, xr.full_like(cookie.isel(rlat=-1), np.nan)], dim="rlat"
            )

    return cookie


def load_environments(envdir, now):
    """
    loads the environment for the given time
    """
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

    # cookie = cookie.drop_vars(["lat", "lon"])
    env = env.drop_dims(["bnds", "level1", "windsector", "soil1"])

    env = env.drop_vars(
        ["P", "wbtemp_13c", "rotated_pole", "height_2m", "height_10m", "height_toa"]
    )

    # convert pressures hPa
    env["pressure"] = env["pressure"] / 100
    env["pressure"].attrs = {
        "units": "hPa",
        "long_name": "pressure",
        "positive": "down",
    }

    env["PS"] = env["PS"] / 100
    env["PS"].attrs = {"units": "hPa", "long_name": "surface pressure"}

    env["PMSL"] = env["PMSL"] / 100
    env["PMSL"].attrs = {"units": "hPa", "long_name": "mean sea level pressure"}

    env = env.squeeze()

    return env


def load_cells(trackdir, day):
    """
    loads the cells for the given day
    """
    with open(os.path.join(trackdir, f"cell_tracks_{day}.json"), "r") as f:
        track_data = json.load(f)

    cells = track_data["cell_data"]

    for cell in cells:
        cell["datelist"] = list(pd.to_datetime(cell["datelist"]))

    return cells


def add_genesis(cell, n_timesteps=6):
    """
    add n_timesteps number of timesteps backwards extrapolation to the cell to aid in investigating cell genesis
    """
    # calculate the mean propagation vector
    u_storm = np.nanmean(cell["delta_x"][1::])
    v_storm = np.nanmean(cell["delta_y"][1::])

    cell["start_date"] = cell["datelist"][0]

    for i in range(n_timesteps):
        # add the genesis time to the datelist
        cell["datelist"].insert(0, cell["datelist"][0] - pd.Timedelta(minutes=5))
        # add the genesis position to the cell
        cell["mass_center_x"].insert(0, cell["mass_center_x"][0] - u_storm)
        cell["mass_center_y"].insert(0, cell["mass_center_y"][0] - v_storm)
        # add the genesis delta_x and delta_y to the cell
        cell["delta_x"].insert(0, u_storm)
        cell["delta_y"].insert(0, v_storm)
        cell["max_val"].insert(0, cell["max_val"][0])
        cell["lat"].insert(0, cell["lat"][0])
        cell["lon"].insert(0, cell["lon"][0])

    return cell


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
