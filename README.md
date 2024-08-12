# PC Rasterize: Rasterize Point Clouds
---

### How to use:

```python
import pc_rasterize as pcr
import glob

files = sorted(glob.glob("../data/points/*.laz"))
# Create a GeoBox grid specification with a 100m buffer around data
geobox = pcr.build_geobox(files, resolution=0.50, crs="5070", buffer=100)
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
# Use rioxarray to save to disk
chm.rio.to_raster("points_chm.tiff")
```
