from collections import namedtuple
from pathlib import Path

import dask.array as da
import geopandas as gpd
import numba as nb
import numpy as np
import pandas as pd
import pdal
import rasterio as rio
import rioxarray as xrio  # noqa: F401
import shapely
import xarray as xr
from odc.geo.geobox import GeoBox, GeoboxTiles
from pdal import Pipeline, Stage

from pc_rasterize._version import __version__  # noqa

__all__ = [
    "build_geobox",
    "filter_files_by_geom",
    "get_file_quickinfo",
    "load_bboxes_from_files",
    "rasterize",
]


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


def _path_to_pipe(path, nthreads=1):
    reader = pdal.Reader(path)
    if nthreads is not None and nthreads > 0:
        # Don't want to over-allocate threads and cause them all to starve
        reader._options["threads"] = nthreads
    return reader.pipeline()


def get_file_quickinfo(path):
    """Get header information from a file. Does not load any data.

    The resulting `namedtuple` has the following fields:

    * n_points : The number of points in the file.
    * dims : The data fields in the file.
    * crs : The CRS object for the file
    * vert_units : The z units for the file's data
    * bounds_2d : (minx, miny, maxx, maxy)
    * bounds_3d : (minx, miny, minz, maxx, maxy, maxz)
    * bbox : Bounding box polygon for the file.

    """
    if isinstance(path, (str, Path)):
        pipe = _path_to_pipe(str(path))
    else:
        raise TypeError("path must be a string or Path object")

    qinfo = pipe.quickinfo
    info = qinfo[list(qinfo)[0]]
    dims = tuple(d.strip() for d in info["dimensions"].split(","))
    n_points = info["num_points"]
    
    # Handle missing SRS metadata
    crs = None
    vert_units = None
    
    srs_info = info.get("srs", {})
    if srs_info:
        # Try to extract CRS information
        srs_json = srs_info.get("json")
        if srs_json:
            try:
                crs = rio.CRS.from_user_input(srs_json)
            except (ValueError, TypeError, rio.errors.CRSError):
                # If CRS parsing fails, leave as None
                pass
        
        # Try to extract vertical units
        units_info = srs_info.get("units", {})
        if units_info:
            vert_units = units_info.get("vertical")

    bounds_2d = tuple(
        info["bounds"][k] for k in ("minx", "miny", "maxx", "maxy")
    )
    bounds_3d = tuple(
        info["bounds"][k]
        for k in ("minx", "miny", "minz", "maxx", "maxy", "maxz")
    )
    bbox = shapely.geometry.box(*bounds_2d)
    return LidarInfo(
        n_points, dims, crs, vert_units, bounds_2d, bounds_3d, bbox
    )


def series_to_array(func):
    def wrapper(x):
        if isinstance(x, pd.Series):
            return func(x.to_numpy())
        return func(x)

    return wrapper


@series_to_array
@nb.jit(nopython=True, nogil=True)
def _asm_agg(x):
    counts = {}
    n = 0
    for v in x:
        if v in x:
            counts[v] += 1
        else:
            counts[v] = 1
        n += 1
    asm = 0.0
    if n > 0:
        n_inv = 1.0 / n
        for c in counts.values():
            p = c * n_inv
            asm += p * p
    return asm


@series_to_array
@nb.jit(nopython=True, nogil=True)
def _entropy_agg(x):
    counts = {}
    n = 0
    for v in x:
        if v in counts:
            counts[v] += 1
        else:
            counts[v] = 1
        n += 1
    entropy = 0.0
    if n > 0:
        n_inv = 1.0 / n
        for cnt in counts.values():
            p = cnt * n_inv
            entropy -= p * np.log(p)
    return entropy


@series_to_array
@nb.jit(nopython=True, nogil=True)
def _cv_agg(x):
    # coefficient of variation
    count = len(x)
    s = np.sum(x)
    ss = np.sum(x * x)
    sd = np.sqrt((ss - ((s * s) / count)) / count)
    m = s / count
    return sd / m


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


def _warp_bboxes_conservative(bboxes, dest_crs, p_detail=None, p_buffer=None):
    if p_buffer is None:
        p_buffer = 0.05
    # Split each side of boxes into many lines. What are lines in the current
    # CRS could become curves in the destination CRS. This step helps to
    # preserve the curvature after the transformation.
    bboxes = _increase_bboxes_detail(bboxes, p=p_detail)
    # Warp
    warped_bboxes = bboxes.to_crs(dest_crs)
    # Add a small buffer to the warped boxes. Line segments can only
    # approximate curves so far. This should add enough margin to cover the
    # formerly enclosed space.
    return warped_bboxes.buffer(_calculate_buffers(warped_bboxes, p=p_buffer))


def _flat_index(x, y, affine, shape):
    *_, nr, nc = shape
    col, row = (~affine) * (x, y)
    row = np.floor(row).astype(int)
    col = np.floor(col).astype(int)
    # TODO: this is a hack to prevent errors but causes double counting of
    # points that fall on the bounding box edges. A more involved fix needs to
    # be implemented.
    row[row == nr] = nr - 1
    col[col == nc] = nc - 1
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


def _chunksize_2d_from_dtype(dtype):
    return da.empty((40_000, 40_000), dtype=dtype).chunksize


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
        pipe = _path_to_pipe(paths[0])
        info = get_file_quickinfo(paths[0])
        if info.crs != dest_crs:
            pipe |= Stage(
                type="filters.reprojection", out_srs=dest_crs.to_wkt()
            )
        pipe = _crop_pipe(pipe, dest_bbox)
        return pipe

    infos = [get_file_quickinfo(p) for p in paths]
    homogeneous_crs = all(infos[0].crs == i.crs for i in infos)
    pipe = Pipeline()
    for p in paths:
        pipe |= _path_to_pipe(p)
        if not homogeneous_crs:
            pipe |= Stage(
                type="filters.reprojection", out_srs=dest_crs.to_wkt()
            )
    pipe = _merge_pipe(pipe)
    if homogeneous_crs:
        pipe |= Stage(type="filters.reprojection", out_srs=dest_crs.to_wkt())
    return _crop_pipe(pipe, dest_bbox)


def _divide_geobox(geobox, chunksize_or_tiles):
    if isinstance(chunksize_or_tiles, GeoboxTiles):
        tiles = chunksize_or_tiles
    else:
        tiles = GeoboxTiles(geobox, tile_shape=chunksize_or_tiles)
    geoboxes = np.empty(tuple(tiles.shape), dtype=object)
    for ind in np.ndindex(tuple(tiles.shape)):
        geoboxes[ind] = tiles.crop[ind].base
    return geoboxes


PDAL_CHUNKSIZE = 100_000


def _load_pc_data(paths, dest_bbox, dest_crs, pdal_filters):
    # Load points that fall within this raster box
    pipe = _build_warped_merged_cropped_pipeline(paths, dest_bbox, dest_crs)
    if pdal_filters:
        for f in pdal_filters:
            pipe |= Stage(**f)
    if not pipe.streamable:
        n = pipe.execute()
        if n == 0:
            return pd.DataFrame()
        return pd.DataFrame(np.concatenate(list(pipe.arrays)))
    # Streaming data
    arrays = list(pipe.iterator(chunk_size=PDAL_CHUNKSIZE))
    if not arrays:
        return pd.DataFrame()
    return pd.DataFrame(np.concatenate(arrays))


def _rasterize(
    geobox, paths, agg_func, agg_func_args, nodata, pdal_filters, dtype
):
    dest_crs = geobox.crs
    affine = geobox.affine
    dest_bbox = geobox.extent.geom
    shape = tuple(geobox.shape)
    if not paths:
        return np.full(shape, nodata, dtype=dtype)

    pts_df = _load_pc_data(paths, dest_bbox, dest_crs, pdal_filters)
    if not len(pts_df):
        return np.full(shape, nodata, dtype=dtype)

    # Use copy to silence pandas chained assignment warning. It doesn't apply
    # here
    pts_df = pts_df[["X", "Y", "Z"]].copy()
    # Bin each point to a pixel location
    pts_df["_bin_"] = _flat_index(
        pts_df.X.to_numpy(), pts_df.Y.to_numpy(), affine, shape
    )

    z_agg = pts_df.groupby("_bin_").Z.agg(agg_func, *agg_func_args)
    grid_flat = np.full(np.prod(shape), nodata, dtype=dtype)
    grid_flat[z_agg.index.to_numpy()] = z_agg.to_numpy()
    return grid_flat.reshape(shape)


_MAX_RECURSE_LEVEL = 3
_N_POINTS_THRESHOLD = 2_000_000


def _rasterize_chunk(
    geobox,
    paths,
    agg_func="max",
    agg_func_args=(),
    nodata=np.nan,
    pdal_filters=(),
    block_info=None,
    max_rlevel=0,
    _rlevel=0,
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

    shape = tuple(geobox.shape)
    assert shape == block_info[None]["chunk-shape"]
    dtype = block_info[None]["dtype"]
    if not paths:
        return np.full(shape, nodata, dtype=dtype)

    expected_points = sum(get_file_quickinfo(p).n_points for p in paths)
    if (
        expected_points > _N_POINTS_THRESHOLD
        and _rlevel < max_rlevel
        and not any(d // 2 == 0 for d in shape)
    ):
        # Break the chunk into 4 sub-chunks and recurse into each one.
        rc = shape[0] // 2
        cc = shape[1] // 2
        chunks = ((rc, shape[0] - rc), (cc, shape[1] - cc))
        tiles = GeoboxTiles(geobox, tile_shape=chunks)
        geoboxes = _divide_geobox(geobox, tiles)
        binned_paths = _bin_files_to_tiles(paths, tiles, geobox.crs)
        idxs = list(np.ndindex(geoboxes.shape))
        result = [[], []]
        for idx in idxs:
            sub_geobox = geoboxes[idx]
            sub_paths = binned_paths[idx]
            sub_block_info = {
                None: {"chunk-shape": sub_geobox.shape, "dtype": dtype}
            }
            result[idx[0]].append(
                # INFO: Recursion
                _rasterize_chunk(
                    sub_geobox,
                    sub_paths,
                    agg_func=agg_func,
                    agg_func_args=agg_func_args,
                    nodata=nodata,
                    pdal_filters=pdal_filters,
                    block_info=sub_block_info,
                    _rlevel=_rlevel + 1,
                )
            )
        # Stitch the results into the full sized chunk
        return np.block(result)
    else:
        return _rasterize(
            geobox,
            paths,
            agg_func=agg_func,
            agg_func_args=agg_func_args,
            nodata=nodata,
            pdal_filters=pdal_filters,
            dtype=dtype,
        )


def _homogenize_crs(infos):
    if len(infos) == 0:
        return infos
    if not all(infos[0].crs == i.crs for i in infos):
        crs = infos[0].crs
        new_infos = [infos[0]]
        for info in infos[1:]:
            if info.crs != crs:
                bbox = gpd.GeoSeries([info.bbox], crs=info.crs)
                warped_bbox = _warp_bboxes_conservative(bbox, crs)
                new_bounds_2d = warped_bbox.total_bounds
                new_bbox = shapely.geometry.box(*new_bounds_2d)
                minx, miny, maxx, maxy = new_bounds_2d
                new_bounds_3d = (
                    minx,
                    miny,
                    info.bounds_3d[2],
                    maxx,
                    maxy,
                    info.bounds_3d[5],
                )
            new_infos.append(
                LidarInfo(
                    info.n_points,
                    info.dims,
                    crs,
                    info.vert_units,
                    new_bounds_2d,
                    new_bounds_3d,
                    new_bbox,
                )
            )
        infos = new_infos
    return infos


def _get_homogeneous_infos(paths):
    infos = [get_file_quickinfo(p) for p in paths]
    # Make sure that all bounding boxes are in the same CRS
    return _homogenize_crs(infos)


def load_bboxes_from_files(paths):
    if isinstance(paths, str):
        paths = [paths]
    infos = _get_homogeneous_infos(paths)
    return gpd.GeoSeries([i.bbox for i in infos], crs=infos[0].crs)


def _bin_files_to_tiles(paths, tiles, dest_crs):
    infos = _get_homogeneous_infos(paths)
    src_crs = infos[0].crs

    tiles_shape = tuple(tiles.shape)
    tile_2d_indices = list(np.ndindex(tiles_shape))
    tile_geoms = [tiles.crop[idx].base.extent.geom for idx in tile_2d_indices]
    tile_bboxes = gpd.GeoSeries(tile_geoms, crs=dest_crs).to_frame("geometry")
    tile_bboxes["tile_idx"] = tile_2d_indices

    src_bboxes = gpd.GeoSeries([i.bbox for i in infos], crs=src_crs)
    if src_crs != dest_crs:
        src_bboxes_warped_to_dest = _warp_bboxes_conservative(
            src_bboxes, dest_crs
        )
        data_bboxes_in_dest = src_bboxes_warped_to_dest.to_frame("geometry")
    else:
        data_bboxes_in_dest = src_bboxes.to_frame("geometry")
    data_bboxes_in_dest["path_idx"] = np.arange(len(paths))

    bins = np.empty(tiles_shape, dtype=object)
    for idx in tile_2d_indices:
        bins[idx] = []
    matches = tile_bboxes.sjoin(data_bboxes_in_dest).sort_values("path_idx")
    for pi, mdf in matches.groupby("path_idx"):
        path = paths[pi]
        for row in mdf.itertuples():
            bins[row.tile_idx].append(path)
    return bins


def _validate_filter_dict(f):
    if "type" not in f or not f["type"].startswith("filters."):
        raise ValueError("Filter objects must have type 'filters.*'")


def _normalize_pdal_filters(filters):
    if not isinstance(filters, (list, tuple)):
        raise TypeError("pdal filters must be a list or tuple")

    normd_filters = []
    for filt in filters:
        if isinstance(filt, Stage):
            filt = filt.options
            _validate_filter_dict(filt)
            normd_filters.append(filt)
        elif isinstance(filt, dict):
            _validate_filter_dict(filt)
            normd_filters.append(filt)
        elif isinstance(filt, Pipeline):
            normd_filters.extend(_normalize_pdal_filters(filt.stages))
        else:
            raise TypeError(
                "filters must contain only dict, Stage, or Pipeline objects"
            )
    return normd_filters


_DEFAULT_AGG_FUNCS = {
    # Pandas built-in funcs
    "count": "count",
    "cov": "cov",
    "max": "max",
    "mean": "mean",
    "median": "median",
    "min": "min",
    "nunique": "nunique",
    "sem": "sem",
    "skew": "skew",
    "std": "std",
    "sum": "sum",
    "quantile": "quantile",
    "var": "var",
    # Custom funcs
    "asm": _asm_agg,
    "cv": _cv_agg,
    "entropy": _entropy_agg,
    "range": lambda x: x.max() - x.min(),
}


def rasterize(
    paths,
    like: GeoBox,
    cell_func="max",
    cell_func_args=(),
    dtype=np.float32,
    chunksize=None,
    nodata=np.nan,
    pdal_filters=(),
    memory_throttling=0,
):
    """Rasterize point cloud files to a given grid specification.

    Parameters:
    -----------
    paths : str or list of str
        The paths of point cloud files to rasterize.
    like : GeoBox
        The grid specification to build a raster from. See the helper function
        :ref:`build_geobox`.
    cell_func : str, callable, optional
        The function to use when aggregating the points that fall within a
        cell. If a `str`, the corresponding builtin function below is used:

        'count'
            Returns the number of points in a cell.
        'cov'
            Returns the covariance of the z values in a cell.
        'max'
            Returns the maximum z value in a cell. (Default)
        'mean'
            Returns the mean z value in a cell.
        'median'
            Returns the median z value in a cell.
        'min'
            Returns the minimum z value in a cell.
        'nunique'
            Returns the number of unique z values in a cell.
        'sem'
            Returns the unbiased standard error of the mean of the z values in
            a cell.
        'skew'
            Returns the skew of the z values in a cell.
        'std'
            Returns the standard deviation of the z values in a cell.
        'sum'
            Returns the sum of the z values in a cell.
        'quantile'
            Returns the quantile value of the z values in a cell. Takes 1
            argument which is a float in the range [0, 1] that sets the
            quantile. The default is ``0.5``.
        'var'
            Returns the variance of the z values in a cell.
        'asm'
            Returns the angular second moment of the z values in a cell.
        'cv'
            Returns the coefficient of variation of the z values in a cell.
        'entropy'
            Returns the entropy of the z values in a cell.
        'range'
            Returns the difference between the max and min z values in a cell.

        If a callable/function is provided, it is applied directly to the
        points within each cell. The function must take a single
        `pandas.Series` object and return a single scalar. The default is the
        `'max'` function.

        Additional arguments can be passed in using the `cell_func_args`
        keyword.
    cell_func_args : tuple, optional
        A tuple of additional arguments to be passed to `cell_func`. The
        default is to pass no additional arguments.
    dtype : str, numpy.dtype, optional
        The dtype to use for the result.
    chunksize : int, 2-tuple of int, optional
        The dask chunksize to use when computing the raster. The default is to
        let dask decide based on the dtype.
    nodata : scalar, optional
        The value to use for cells with no points. Default is NaN.
    pdal_filters : tuple or list of {pdal.Stage, pdal.Pipeline, dict}, optional
        A tuple or list of PDAL filters to be applied when loading the point
        cloud. See PDAL's
        `filter documentation <https://pdal.io/en/latest/stages/filters.html>`_
        for more information. Default is to apply no additional filters.
    memory_throttling : int, optional
        The amount by which memory usage should be throttled for the Dask
        workers. Acceptible values are ``0``, ``1``, ``2``, and ``3``.
        A value of ``0`` does not throttle memory at all. This results in
        potentially large amounts of memory being used but also achieves the
        greatest speed. Higher values cause memory to be conserved more. A
        value of ``3`` very tightly restricts the amount of memory that can be
        used by each worker but causes the longest running time. If your
        `rasterize` task runs out of memory, try adjusting this number up. This
        option is especially helpful for systems with a large number of CPU
        cores but small amount of system memory. The default is ``0``.

    Returns
    -------
    raster : xarray.DataArray
        The resulting raster as an `xarray.DataArray` object.

    """
    if cell_func is None:
        agg_func = _DEFAULT_AGG_FUNCS["max"]
    elif callable(cell_func):
        agg_func = cell_func
    elif cell_func in _DEFAULT_AGG_FUNCS:
        agg_func = _DEFAULT_AGG_FUNCS[cell_func]
    else:
        raise TypeError(
            "cell_func must be callable or one of the listed reduction "
            "functions."
        )
    if not isinstance(cell_func_args, (list, tuple)):
        raise TypeError("cell_func_args must be a tuple or list")
    if memory_throttling < 0 or memory_throttling > _MAX_RECURSE_LEVEL:
        raise ValueError(
            "memory_throttling value must be between 0 and "
            f"{_MAX_RECURSE_LEVEL}, inclusive."
        )
    pdal_filters = _normalize_pdal_filters(pdal_filters)
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
        chunksize = _chunksize_2d_from_dtype(dtype)
    tiles = GeoboxTiles(like, tile_shape=chunksize)
    chunks = tiles.chunks
    geoboxes = _divide_geobox(like, tiles)
    geoboxes = da.from_array(geoboxes, chunks=1)

    # Array with dims equal to the number of tiles in each dim (y, x). Each
    # element is a list of file paths that intersected the corresponding tile.
    binned_paths = _bin_files_to_tiles(paths, tiles, crs)
    binned_paths = da.from_array(binned_paths, chunks=1)

    data = da.map_blocks(
        _rasterize_chunk,
        geoboxes,
        binned_paths,
        agg_func=agg_func,
        agg_func_args=cell_func_args,
        nodata=nodata,
        pdal_filters=pdal_filters,
        chunks=chunks,
        meta=np.array((), dtype=dtype),
        max_rlevel=memory_throttling,
    )
    return (
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
        .rio.write_crs(crs)
    )


def build_geobox(paths, resolution, crs=None, buffer=0):
    """Helper function for building a GeoBox object to specify a grid.

    This helps build a `GeoBox` that can be used with :ref:`rasterize`.

    Parameters
    ----------
    paths : str or list of str
        The point cloud files to build a grid for.
    resolution : int
        The resolution in `crs` units for the grid.
    crs : int, str, rasterio.CRS, optional
        The CRS to use for the grid. If left blank, the CRS from the point
        cloud files is used.
    buffer : int, float, optional
        The distance in `crs` units to buffer or expand the grid from the
        bounding box around the point cloud data. The default is 0.

    Returns:
    --------
    geobox : Geobox
        The resulting grid specification.

    Notes:
    ------
    **Known Issue**: When point cloud files lack CRS metadata and crs=None is 
    passed, GeoBox.from_bbox() defaults to EPSG:4326. This can create invalid 
    geometries when coordinates are in projected systems (e.g., UTM coordinates 
    being treated as lat/lon). 
    
    **Potential Solutions**:
    1. Require explicit CRS: Raise error when both file CRS and user CRS are None
    2. Coordinate-based CRS detection: Guess appropriate CRS from coordinate ranges  
    3. Default to reasonable projected CRS: Use sensible default for coordinate range
    
    For files without CRS metadata, explicitly specify the crs parameter.

    """
    if isinstance(paths, str):
        paths = [paths]
    if resolution <= 0:
        raise ValueError("resolution must be a positive scalar")

    infos = _get_homogeneous_infos(paths)
    boxes = gpd.GeoSeries([i.bbox for i in infos], crs=infos[0].crs)
    if crs is not None:
        target_crs = crs
        if infos[0].crs is not None:
            # File has CRS metadata - transform to target CRS
            boxes = boxes.to_crs(crs)
        else:
            # File has no CRS metadata - assume coordinates are already in target CRS
            boxes = boxes.set_crs(crs)
    else:
        target_crs = infos[0].crs
    if buffer:
        boxes = boxes.buffer(buffer)
    bbox = shapely.geometry.box(*boxes.total_bounds)
    return GeoBox.from_bbox(
        bbox=bbox.bounds, crs=target_crs, resolution=resolution
    )


def filter_files_by_geom(files, filter_geom, filter_geom_crs):
    filter_geom_crs = rio.CRS.from_user_input(filter_geom_crs)
    finfos = _get_homogeneous_infos(files)
    files_crs = finfos[0].crs
    file_geoms = gpd.GeoSeries([fi.bbox for fi in finfos], crs=files_crs)
    if files_crs != filter_geom_crs:
        file_geoms = _warp_bboxes_conservative(file_geoms, filter_geom_crs)
    file_geoms = gpd.GeoDataFrame(
        {"file": files, "geometry": file_geoms}, crs=filter_geom_crs
    )
    filter_df = gpd.GeoDataFrame(
        {"geometry": [filter_geom]}, crs=filter_geom_crs
    )
    match = filter_df.sjoin(file_geoms)
    return sorted(match.file.to_list())
