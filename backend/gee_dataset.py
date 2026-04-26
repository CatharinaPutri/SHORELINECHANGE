"""
==========================================================
  Google Earth Engine — Dataset Generation Module
  Sentinel-2 SR Harmonized Composite with Spectral Indices
  Supports:
  - preview tile
  - export to Google Drive
  - direct download to local GeoTIFF
==========================================================
"""

from __future__ import annotations

import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Dict

import ee

_gee_initialized = False


def initialize_gee(project_id: str = "") -> str:
    """Authenticate and initialize GEE."""
    global _gee_initialized

    if _gee_initialized:
        return "GEE already initialized."

    try:
        if project_id:
            ee.Initialize(project=project_id)
        else:
            ee.Authenticate()
            ee.Initialize()
        _gee_initialized = True
        return "GEE initialized successfully."
    except Exception:
        ee.Authenticate()
        if project_id:
            ee.Initialize(project=project_id)
        else:
            ee.Initialize()
        _gee_initialized = True
        return "GEE authenticated and initialized."


def _to_ee_geometry(roi_geojson: Dict[str, Any]) -> ee.Geometry:
    if not isinstance(roi_geojson, dict):
        raise ValueError("ROI must be a valid GeoJSON dictionary.")

    geo_type = roi_geojson.get("type")

    if geo_type == "Feature":
        geometry = roi_geojson.get("geometry")
        if not geometry:
            raise ValueError("GeoJSON Feature does not contain a geometry.")
        return ee.Geometry(geometry)

    if geo_type == "FeatureCollection":
        features = roi_geojson.get("features", [])
        if not features:
            raise ValueError("GeoJSON FeatureCollection is empty.")
        return ee.FeatureCollection(roi_geojson).geometry()

    if geo_type in {
        "Polygon",
        "MultiPolygon",
        "Point",
        "MultiPoint",
        "LineString",
        "MultiLineString",
        "LinearRing",
        "GeometryCollection",
    }:
        return ee.Geometry(roi_geojson)

    raise ValueError(f"Unsupported ROI GeoJSON type: {geo_type}")


def _mask_s2_clouds(image: ee.Image) -> ee.Image:
    """Mask clouds using Sentinel-2 QA60 band."""
    qa = image.select("QA60")
    cloud_bit = 1 << 10
    cirrus_bit = 1 << 11

    mask = qa.bitwiseAnd(cloud_bit).eq(0).And(
        qa.bitwiseAnd(cirrus_bit).eq(0)
    )

    return (
        image.updateMask(mask)
        .divide(10000)
        .copyProperties(image, image.propertyNames())
    )


def generate_composite(
    roi_geojson: Dict[str, Any],
    date_start: str,
    date_end: str,
    cloud_max: int = 15,
) -> Dict[str, Any]:
    """
    Generate a 7-band Sentinel-2 composite:
    Red, Green, Blue, NIR, MNDWI, NDVI, NDBI
    """
    roi = _to_ee_geometry(roi_geojson)

    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterDate(date_start, date_end)
        .filterBounds(roi)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_max))
        .map(_mask_s2_clouds)
    )

    image_count = int(collection.size().getInfo())
    if image_count == 0:
        raise ValueError(
            "No Sentinel-2 images found for the selected ROI/date range/cloud threshold."
        )

    composite = collection.median().clip(roi)

    red = composite.select("B4").rename("Red")
    green = composite.select("B3").rename("Green")
    blue = composite.select("B2").rename("Blue")
    nir = composite.select("B8").rename("NIR")

    mndwi = composite.normalizedDifference(["B3", "B11"]).rename("MNDWI")
    ndvi = composite.normalizedDifference(["B8", "B4"]).rename("NDVI")
    ndbi = composite.normalizedDifference(["B11", "B8"]).rename("NDBI")

    stacked = (
        red.addBands(green)
        .addBands(blue)
        .addBands(nir)
        .addBands(mndwi)
        .addBands(ndvi)
        .addBands(ndbi)
    )

    return {
        "composite_image": stacked,
        "composite_rgb": composite,
        "image_count": image_count,
        "bands": ["Red", "Green", "Blue", "NIR", "MNDWI", "NDVI", "NDBI"],
        "roi": roi,
    }


def get_composite_tile_url(stacked_image: ee.Image) -> str:
    """Get a tile URL for Leaflet visualization."""
    vis_params = {
        "bands": ["Red", "Green", "Blue"],
        "min": 0.0,
        "max": 0.3,
    }
    map_id = stacked_image.getMapId(vis_params)
    return map_id["tile_fetcher"].url_format


def export_composite_to_drive(
    composite_image: ee.Image,
    roi_geojson: Dict[str, Any],
    folder: str = "GEE_Shoreline_Export",
    file_prefix: str = "S2_Composite",
    scale: int = 10,
    crs: str = "EPSG:32749",
) -> str:
    """Start an export task to Google Drive."""
    roi = _to_ee_geometry(roi_geojson)

    task = ee.batch.Export.image.toDrive(
        image=composite_image,
        description=file_prefix,
        folder=folder,
        fileNamePrefix=file_prefix,
        region=roi,
        scale=scale,
        crs=crs,
        maxPixels=1e13,
        fileFormat="GeoTIFF",
    )
    task.start()
    return task.id


def download_composite_to_local(
    roi_geojson: Dict[str, Any],
    date_start: str,
    date_end: str,
    out_tif_path: str,
    cloud_max: int = 15,
    scale: int = 10,
    crs: str = "EPSG:32749",
) -> Dict[str, Any]:
    """
    Directly download the GEE composite to a local GeoTIFF file.

    Notes
    -----
    This route is suitable for moderate ROI sizes.
    For very large ROI / raster sizes, use Export to Drive instead.
    """
    composite_info = generate_composite(
        roi_geojson=roi_geojson,
        date_start=date_start,
        date_end=date_end,
        cloud_max=cloud_max,
    )

    stacked = composite_info["composite_image"]
    roi_geom = _to_ee_geometry(roi_geojson)

    out_path = Path(out_tif_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Earth Engine direct download. Region is passed as GeoJSON geometry.
    download_url = stacked.getDownloadURL({
        "name": out_path.stem,
        "region": roi_geom.getInfo(),
        "scale": scale,
        "crs": crs,
        "format": "GEO_TIFF",
        "filePerBand": False,
    })

    temp_download = out_path.with_suffix(".download")
    with urllib.request.urlopen(download_url) as response, open(temp_download, "wb") as f:
        shutil.copyfileobj(response, f)

    with open(temp_download, "rb") as f:
        signature = f.read(4)

    # Some GEE downloads may still come zipped.
    if signature == b"PK\x03\x04":
        with zipfile.ZipFile(temp_download, "r") as zf:
            tif_members = [
                name for name in zf.namelist()
                if name.lower().endswith(".tif") or name.lower().endswith(".tiff")
            ]
            if not tif_members:
                raise ValueError("Downloaded archive does not contain a GeoTIFF file.")

            with zf.open(tif_members[0]) as src, open(out_path, "wb") as dst:
                shutil.copyfileobj(src, dst)

        temp_download.unlink(missing_ok=True)
    else:
        temp_download.replace(out_path)

    return {
        "tif_path": str(out_path),
        "image_count": composite_info["image_count"],
        "bands": composite_info["bands"],
    }