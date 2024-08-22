# PC Rasterize: Rasterize Point Clouds in Parallel
---

## How to use:

```python
import pc_rasterize as pcr
import glob

files = sorted(glob.glob("../data/points/*.laz"))
# Create a GeoBox grid specification with a 100m buffer around data
geobox = pcr.build_geobox(files, resolution=0.50, crs="5070", buffer=100)
# Build a lazy CHM raster
chm = pcr.rasterize(
    files,
    geobox,
    cell_func="max",
    # Set custom dask chunk-size
    chunksize=(500, 500),
    nodata=np.nan,
    # Filter out points over 100m
    pdal_filters=[
        {
            "type": "filters.expression",
            "expression": "Z < 100"
        }
    ],
)
```

### Saving with default dask scheduling:

```python
# Use rioxarray to save to disk
chm.rio.to_raster("points_chm.tiff", tiled=True)
```

### Saving with dask's more advanced scheduling:
Dask's more advanced 'distributed' scheduling also provides a dashboard at
[http://localhost:8787/status](http://localhost:8787/status) for viewing
progress in your browser.

```python
from dask.distributed import Client, LocalCluster, Lock

with LocalCluster() as cluster, Client(cluster) as client:
    chm.rio.to_raster("points_chm.tiff", tiled=True, lock=Lock("rio"))
```
