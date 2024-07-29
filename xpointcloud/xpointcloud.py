import json
from collections import namedtuple
from collections.abc import Iterable
from pathlib import Path

import dask.array as da
import geopandas as gpd
import numpy as np
import pandas as pd
import pdal
import rasterio as rio
import rioxarray as xrio  # noqa: F401
import shapely
import xarray as xr
from odc.geo.geobox import GeoBox, GeoboxTiles
from pdal import Pipeline, Stage


def pdumps(obj):
    return json.dumps([obj])


LidarInfo = namedtuple(
    "LidarInfo",
    (
        "n_points",
        "dims",
        "crs",
        "vert_units",
        "bounds_2d",
        "bounds_3d",
        "bbox",
    ),
)


def get_quickinfo(paths_or_pipeline):
    if isinstance(paths_or_pipeline, (str, Path)):
        pl = pdal.Pipeline(pdumps(str(paths_or_pipeline)))
    elif isinstance(paths_or_pipeline, Iterable):
        pl = _build_input_pipeline(paths_or_pipeline)
    elif isinstance(paths_or_pipeline, pdal.Pipeline):
        pl = paths_or_pipeline
    else:
        raise TypeError("input must be a file path or pipeline object")
    qinfo = pl.quickinfo
    qinfos = qinfo[list(qinfo)[0]]
    if isinstance(qinfos, dict):
        qinfos = [qinfos]
    info_objs = []
    for info in qinfos:
        dims = tuple(d.strip() for d in info["dimensions"].split(","))
        n_points = info["num_points"]
        crs = rio.CRS.from_user_input(info["srs"]["json"])
        vert_units = info["srs"]["units"]["vertical"]
        bounds_2d = tuple(
            info["bounds"][k] for k in ("minx", "miny", "maxx", "maxy")
        )
        bounds_3d = tuple(
            info["bounds"][k]
            for k in ("minx", "miny", "minz", "maxx", "maxy", "maxz")
        )
        bbox = shapely.geometry.box(*bounds_2d)
        info_objs.append(
            LidarInfo(
                n_points, dims, crs, vert_units, bounds_2d, bounds_3d, bbox
            )
        )
    return info_objs


def get_bboxes(paths):
    infos = get_quickinfo(paths)
    return gpd.GeoSeries([i.bbox for i in infos], crs=infos[0].crs)


def _geoms_to_bboxes(geoms):
    return gpd.GeoSeries(
        [
            shapely.geometry.box(b.minx, b.miny, b.maxx, b.maxy)
            for b in geoms.bounds.itertuples()
        ],
        crs=geoms.crs,
    )


def _calculate_buffers(boxes, p=0.05):
    bdf = boxes.bounds
    bdf["xspan"] = bdf.maxx - bdf.minx
    bdf["yspan"] = bdf.maxy - bdf.miny
    min_spans = bdf[["xspan", "yspan"]].min(axis=1)
    return (min_spans * p).to_numpy()


def _calculate_seg_len(bounds, p=0.01):
    minx, miny, maxx, maxy = bounds
    min_span = min(maxx - minx, maxy - miny)
    return p * min_span


def _increase_bboxes_detail(bboxes, p=None):
    args = (bboxes.total_bounds,) if p is None else (bboxes.total_bounds, p)
    return bboxes.segmentize(_calculate_seg_len(*args))


def _warp_bboxes_conservative(bboxes, dest_crs):
    # Split each side of boxes into many lines. What are lines in the current
    # CRS could become curves in the destination CRS. This step helps to
    # preserve the curvature after the transformation.
    bboxes = _increase_bboxes_detail(bboxes)
    # Warp
    warped_bboxes = bboxes.to_crs(dest_crs)
    # Add a small buffer to the warped boxes. Line segments can only
    # approximate curves so far. This should add enough margin to cover the
    # formerly enclosed space.
    return warped_bboxes.buffer(_calculate_buffers(warped_bboxes))


def build_geobox(paths, resolution, crs=None, buffer=None):
    if isinstance(paths, str):
        paths = [paths]
    if resolution < 0:
        raise ValueError("resolution must be a positive scalar")

    infos = get_quickinfo(paths)
    boxes = get_bboxes(paths)
    if crs is not None:
        target_crs = crs
        boxes = boxes.to_crs(crs)
    else:
        target_crs = infos[0].crs
    if buffer is not None:
        boxes = boxes.buffer(buffer)
    bbox = shapely.geometry.box(*boxes.total_bounds)
    return GeoBox.from_bbox(
        bbox=bbox.bounds, crs=target_crs, resolution=resolution
    )


def execute(pipeline):
    try:
        _ = pipeline.metadata
    except RuntimeError:
        pipeline.execute()
    return pipeline


def load(*paths):
    paths = [p if isinstance(p, str) else str(p) for p in paths]
    pl = pdal.Pipeline()
    for p in paths:
        pl |= pdal.Pipeline(pdumps(p))
    if len(paths) > 1:
        pl |= pdal.Stage(type="filters.merge")
    return execute(pl)


def load_to_dataframe(*paths):
    pipeline = load(*paths)
    return pd.concat([pd.DataFrame(arr) for arr in pipeline.arrays])


def pdal_df_to_gdf(pts_df, crs=None):
    x = pts_df["X"]
    y = pts_df["Y"]
    z = pts_df["Z"]
    return gpd.GeoDataFrame(
        pts_df, geometry=gpd.points_from_xy(x, y, z, crs=crs)
    )


def load_to_geodataframe(*paths):
    pipeline = load(*paths)
    crs = rio.CRS.from_user_input(pipeline.srswkt2)
    pts_df = pd.concat([pd.DataFrame(arr) for arr in pipeline.arrays])
    return pdal_df_to_gdf(pts_df, crs)


def xy_to_rowcol(x, y, affine):
    """
    Convert (x, y) coords to (row, col) index values using the transformation.
    """
    col, row = (~affine) * (x, y)
    row = np.floor(row).astype(int)
    col = np.floor(col).astype(int)
    return row, col


def flat_index(x, y, affine, shape):
    row, col = xy_to_rowcol(x, y, affine)
    return np.ravel_multi_index((row, col), shape)


def _build_x_coord(affine, shape):
    nx = shape[1]
    # Cell size for x dim
    a = affine.a
    tmatrix = np.array(affine).reshape((3, 3))
    xc = (tmatrix @ np.array([np.arange(nx), np.zeros(nx), np.ones(nx)]))[0]
    xc += a / 2
    # Copy to trim off excess base array
    return xc.copy()


def _build_y_coord(affine, shape):
    ny = shape[0]
    # Cell size for y dim (should be < 0)
    e = affine.e
    tmatrix = np.array(affine).reshape((3, 3))
    yc = (tmatrix @ np.array([np.zeros(ny), np.arange(ny), np.ones(ny)]))[1]
    yc += e / 2
    # Copy to trim off excess base array
    return yc.copy()


def chunksize_2d_from_dtype(dtype):
    return da.empty((40_000, 40_000), dtype=dtype).chunksize


def _build_input_pipeline(paths):
    if len(paths) == 1:
        return Pipeline(json.dumps(paths))

    pipe = Pipeline()
    for p in paths:
        pipe |= Pipeline(json.dumps([p]))
    pipe |= Stage(type="filters.merge")
    return pipe


def iqr_med_z_filter(z, k):
    iqr = z.quantile(0.75) - z.quantile(0.25)
    med = z.median()
    zr = (z.max() - z.min()) + 1e-6
    rratio = iqr / zr
    zq = np.abs((z - med) / iqr)
    if iqr > 0 and (rratio < 0.1):
        return zq < k
    return np.ones(len(z), dtype=bool)


def _crop_pipe(pipe, bbox, tag=None):
    minx, miny, maxx, maxy = bbox.bounds
    tag = {} if tag is None else {"tag": tag}
    return pipe | Stage(
        type="filters.crop", bounds=f"([{minx},{maxx}],[{miny},{maxy}])", **tag
    )


def _merge_pipe(pipe, inputs=None, tag=None):
    kwargs = {}
    if inputs is not None:
        inputs = list(inputs)
        kwargs["inputs"] = inputs
    if tag is not None:
        kwargs["tag"] = tag
    return pipe | Stage(type="filters.merge", **kwargs)


def _build_warped_merged_cropped_pipeline(paths, dest_bbox, dest_crs):
    if len(paths) == 1:
        pipe = Pipeline(json.dumps(paths))
        info = get_quickinfo(paths[0])[0]
        if info.crs != dest_crs:
            pipe |= Stage(
                type="filters.reprojection", out_srs=dest_crs.to_wkt()
            )
        pipe = _crop_pipe(pipe, dest_bbox)
        return pipe

    infos = [get_quickinfo(p)[0] for p in paths]
    homogeneous_crs = all(infos[0].crs == i.crs for i in infos)
    pipe = Pipeline()
    for p in paths:
        pipe |= Pipeline(json.dumps([p]))
        if not homogeneous_crs:
            pipe |= Stage(
                type="filters.reprojection", out_srs=dest_crs.to_wkt()
            )
    pipe = _merge_pipe(pipe)
    if homogeneous_crs:
        pipe |= Stage(type="filters.reprojection", out_srs=dest_crs.to_wkt())
    return _crop_pipe(pipe, dest_bbox)


def _rasterize_chunk(
    geobox,
    paths,
    agg_func=None,
    nodata=np.nan,
    robust_filter=False,
    zfilter_func=None,
    block_info=None,
):
    if isinstance(geobox, np.ndarray):
        geobox = geobox.item()
    else:
        assert isinstance(geobox, GeoBox)
    if isinstance(paths, np.ndarray):
        paths = paths.item()
    elif not isinstance(paths, (list, tuple)):
        raise TypeError(
            "Paths input must be a list or tuple. Got: {type(paths)}"
        )
    if geobox.crs is None:
        raise ValueError("No destination CRS specified")

    dest_crs = geobox.crs
    affine = geobox.affine
    shape = tuple(geobox.shape)
    dest_bbox = geobox.extent.geom
    assert shape == block_info[None]["chunk-shape"]
    dtype = block_info[None]["dtype"]
    if len(paths) == 0:
        return np.full(shape, nodata, dtype=dtype)

    # Load points that fall within this chunk
    pipe = _build_warped_merged_cropped_pipeline(paths, dest_bbox, dest_crs)
    n = pipe.execute()
    if n == 0:
        return np.full(shape, nodata, dtype=dtype)

    # Build the dataframe
    pts_df = pd.concat([pd.DataFrame(arr) for arr in pipe.arrays])
    pipe = None
    pts_df = pts_df[["X", "Y", "Z"]]
    if robust_filter:
        pts_df = pts_df[iqr_med_z_filter(pts_df.Z, 10)]
    if zfilter_func is not None:
        pts_df = pts_df[zfilter_func(pts_df.Z.to_numpy())]
    # Bin each point to a pixel location
    pts_df["_bin_"] = flat_index(
        pts_df.X.to_numpy(), pts_df.Y.to_numpy(), affine, shape
    )

    z_agg = pts_df.groupby("_bin_").Z.max()
    grid_flat = np.full(np.prod(shape), nodata, dtype=dtype)
    grid_flat[z_agg.index.to_numpy()] = z_agg.to_numpy()
    return grid_flat.reshape((shape))


def _bin_files_to_tiles(paths, tiles, dest_crs):
    pipe = _build_input_pipeline(paths)
    infos = get_quickinfo(pipe)

    src_crs = infos[0].crs
    tiles_shape = tuple(tiles.shape)

    if src_crs != dest_crs:
        src_bboxes = gpd.GeoSeries([i.bbox for i in infos], crs=src_crs)
        src_bboxes_warped_to_dest = _warp_bboxes_conservative(
            src_bboxes, dest_crs
        )
        data_bboxes_in_dest = src_bboxes_warped_to_dest.to_frame("geometry")
    else:
        data_bboxes_in_dest = src_bboxes.to_frame("geometry")
    data_bboxes_in_dest["path_idx"] = np.arange(len(paths))
    tile_bboxes = gpd.GeoSeries(
        [tiles.crop[idx].base.extent.geom for idx in np.ndindex(tiles_shape)],
        crs=dest_crs,
    ).to_frame("geometry")
    tile_bboxes["tile_idx"] = list(np.ndindex(tiles_shape))

    bins = np.empty(tiles_shape, dtype=object)
    for idx in np.ndindex(tiles_shape):
        bins[idx] = []
    matches = tile_bboxes.sjoin(data_bboxes_in_dest).sort_values("path_idx")
    for pi, mdf in matches.groupby("path_idx"):
        path = paths[pi]
        for row in mdf.itertuples():
            bins[row.tile_idx].append(path)
    return bins


def rasterize(
    paths,
    like: GeoBox,
    dtype=np.float32,
    nodata=np.nan,
    robust_filter=False,
    zfilter_func=None,
    chunksize=None,
):
    if zfilter_func is not None and not callable(zfilter_func):
        raise TypeError("zfilter_func must be callable")
    dtype = np.dtype(dtype)

    if isinstance(paths, str):
        paths = [paths]
    paths = sorted(paths)
    crs = like.crs
    if crs is None:
        raise ValueError("No destination CRS specified")
    affine = like.affine
    shape = tuple(like.shape)
    if chunksize is None:
        chunksize = chunksize_2d_from_dtype(dtype)
    tiles = GeoboxTiles(like, tile_shape=chunksize)
    chunks = tiles.chunks

    geoboxes = np.empty(tuple(tiles.shape), dtype=object)
    for ind in np.ndindex(tuple(tiles.shape)):
        geoboxes[ind] = tiles.crop[ind].base
    geoboxes = da.from_array(geoboxes, chunks=1)

    # Array with dims equal to the number of tiles in each dim (y, x). Each
    # element is a list of file paths that intersected the given tile.
    binned_paths = _bin_files_to_tiles(paths, tiles, crs)
    binned_paths = da.from_array(binned_paths, chunks=1)

    data = da.map_blocks(
        _rasterize_chunk,
        geoboxes,
        binned_paths,
        agg_func=np.max,
        nodata=nodata,
        robust_filter=robust_filter,
        zfilter_func=zfilter_func,
        chunks=chunks,
        meta=np.array((), dtype=dtype),
    )
    raster = (
        xr.DataArray(
            data,
            dims=("y", "x"),
            coords=(
                _build_y_coord(affine, shape),
                _build_x_coord(affine, shape),
            ),
        )
        .rename("rasterized_pc")
        .expand_dims(band=[1])
        .rio.set_spatial_dims(y_dim="y", x_dim="x")
        .rio.write_nodata(dtype.type(nodata))
    )
    if crs is not None:
        raster = raster.rio.write_crs(crs)
    return raster
