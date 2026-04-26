"""
==========================================================
  Random Forest Shoreline Extraction Module
  Direct Model Version
==========================================================
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import geopandas as gpd
import joblib
import numpy as np
import rasterio
from rasterio.windows import Window
from shapely.geometry import LineString
from skimage import measure
from skimage.filters import gaussian
from skimage.morphology import (
    remove_small_objects,
    remove_small_holes,
    binary_opening,
    binary_closing,
    binary_dilation,
    disk,
)

# ---------------------------------------------------------------------------
# Fixed Model Path
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "random_forest_shoreline_baru.pkl"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FEATURES = ["Red", "Green", "Blue", "NIR", "MNDWI", "NDVI", "NDBI"]
N_FEATURES = len(FEATURES)
WATER_LABEL = 0
CHUNK_SIZE = 1024

# Morphological parameters
MIN_OBJECT_SIZE = 64
MIN_HOLE_SIZE = 64
MORPH_RADIUS = 1

# Open sea extraction
SEA_OPEN_RADIUS = 4
SEA_CLOSE_RADIUS = 2
SEA_DILATE_RADIUS = 1
SEA_MIN_AREA = 5000

# Estuary generalization
ESTUARY_CLOSE_RADIUS = 8
ESTUARY_OPEN_RADIUS = 2
ESTUARY_HOLE_AREA = 1500

# Smoothing
GAUSSIAN_SIGMA = 2.0
CHAIKIN_ITERS = 6
LINE_SMOOTH_WINDOW = 9
SIMPLIFY_TOL_M = 2.0
MIN_VERTICES = 15
MIN_LINE_LENGTH_M = 50.0


@lru_cache(maxsize=1)
def _load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model tidak ditemukan: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


def _write_empty_geojson(output_path: str) -> None:
    empty = {"type": "FeatureCollection", "features": []}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(empty, f)


def run_shoreline_inference(
    tif_path: str,
    output_dir: str,
    year_label: str = "unknown",
) -> str:
    """
    Run full shoreline extraction pipeline.

    Output behavior:
    - GPKG disimpan pada CRS asli raster untuk analisis spasial.
    - GeoJSON direproyeksi ke EPSG:4326 agar aman untuk web map / Leaflet.
    - length_m dihitung sebelum reproyeksi, sehingga tetap dalam satuan meter.
    """
    tif_path = str(Path(tif_path).resolve())
    output_dir = str(Path(output_dir).resolve())

    if not os.path.exists(tif_path):
        raise FileNotFoundError(f"Raster tidak ditemukan: {tif_path}")

    os.makedirs(output_dir, exist_ok=True)
    basename = os.path.splitext(os.path.basename(tif_path))[0]

    rf_model = _load_model()

    # 1) Predict raster
    pred_map, valid_mask, transform, crs = _predict_raster(tif_path, rf_model)

    # 2) Water mask
    water_mask = np.zeros_like(pred_map, dtype=bool)
    water_mask[valid_mask] = pred_map[valid_mask] == WATER_LABEL

    # 3) Clean water mask
    water_clean = _postprocess_water_mask(water_mask, valid_mask)

    # 4) Extract open sea
    open_sea = _extract_open_sea(water_clean, valid_mask)

    # 5) Generalize sea mask
    sea_gen = _generalize_sea_mask(open_sea, valid_mask)

    # 6) Extract shoreline vectors
    shoreline_gdf = _extract_shorelines(sea_gen, valid_mask, transform, crs, tif_path)

    # 7) Save outputs
    geojson_path = os.path.join(output_dir, f"{basename}_shoreline.geojson")
    gpkg_path = os.path.join(output_dir, f"{basename}_shoreline.gpkg")

    if shoreline_gdf.empty:
        _write_empty_geojson(geojson_path)
        return geojson_path

    if shoreline_gdf.crs is None:
        raise ValueError(
            "CRS output shoreline tidak tersedia. "
            "Pastikan raster input memiliki CRS yang valid."
        )

    # -----------------------------------------------------------------------
    # Simpan atribut analisis pada CRS asli raster
    # length_m harus dihitung sebelum reproyeksi agar tetap dalam meter
    # -----------------------------------------------------------------------
    shoreline_gdf["year"] = year_label
    shoreline_gdf["length_m"] = shoreline_gdf.geometry.length.round(3)

    # -----------------------------------------------------------------------
    # Simpan hasil analisis utama pada CRS asli
    # Cocok untuk DSAS / analisis spasial lanjutan
    # -----------------------------------------------------------------------
    shoreline_gdf.to_file(gpkg_path, layer="shoreline", driver="GPKG")

    # -----------------------------------------------------------------------
    # Siapkan output khusus web
    # GeoJSON untuk Leaflet harus dalam EPSG:4326
    # -----------------------------------------------------------------------
    shoreline_web = shoreline_gdf.to_crs(epsg=4326)
    shoreline_web.to_file(geojson_path, driver="GeoJSON")

    return geojson_path


def _predict_raster(raster_path, model):
    with rasterio.open(raster_path) as src:
        h, w = src.height, src.width
        transform = src.transform
        crs = src.crs

        pred_map = np.full((h, w), 255, dtype=np.uint8)
        valid_mask = np.zeros((h, w), dtype=bool)

        for row in range(0, h, CHUNK_SIZE):
            for col in range(0, w, CHUNK_SIZE):
                ch = min(CHUNK_SIZE, h - row)
                cw = min(CHUNK_SIZE, w - col)
                window = Window(col, row, cw, ch)

                data = src.read(
                    indexes=list(range(1, N_FEATURES + 1)),
                    window=window,
                    masked=True,
                ).astype(np.float32)

                arr = np.moveaxis(data.filled(np.nan), 0, -1)
                valid = np.all(np.isfinite(arr), axis=2)

                if not np.any(valid):
                    continue

                block_pred = np.full((ch, cw), 255, dtype=np.uint8)
                block_valid = np.zeros((ch, cw), dtype=bool)

                X = arr[valid]
                y_pred = model.predict(X).astype(np.uint8)

                block_pred[valid] = y_pred
                block_valid[valid] = True

                rs = slice(row, row + ch)
                cs = slice(col, col + cw)

                pred_map[rs, cs] = block_pred
                valid_mask[rs, cs] = block_valid

    return pred_map, valid_mask, transform, crs


def _postprocess_water_mask(water_mask, valid_mask):
    mask = water_mask.copy().astype(bool)
    mask[~valid_mask] = False
    mask = remove_small_objects(mask, min_size=MIN_OBJECT_SIZE)
    mask = remove_small_holes(mask, area_threshold=MIN_HOLE_SIZE)

    se = disk(MORPH_RADIUS)
    mask = binary_closing(mask, se)
    mask = binary_opening(mask, se)

    mask[~valid_mask] = False
    return mask


def _get_border_connected(mask):
    labels = measure.label(mask.astype(np.uint8), connectivity=2)

    if labels.max() == 0:
        return np.zeros_like(mask, dtype=bool), labels, np.array([], dtype=np.int32)

    border_labels = np.unique(
        np.concatenate([
            labels[0, :],
            labels[-1, :],
            labels[:, 0],
            labels[:, -1],
        ])
    )
    border_labels = border_labels[border_labels != 0]

    return np.isin(labels, border_labels), labels, border_labels


def _keep_largest(mask):
    labels = measure.label(mask.astype(np.uint8), connectivity=2)

    if labels.max() == 0:
        return np.zeros_like(mask, dtype=bool)

    ids, counts = np.unique(labels[labels > 0], return_counts=True)
    return labels == ids[np.argmax(counts)]


def _extract_open_sea(water_mask, valid_mask):
    candidate = water_mask.copy().astype(bool)
    candidate[~valid_mask] = False

    border_water, _, border_labels = _get_border_connected(candidate)

    if len(border_labels) == 0:
        fallback = _keep_largest(candidate)
        fallback[~valid_mask] = False
        return fallback

    sea = border_water

    if SEA_OPEN_RADIUS > 0:
        sea = binary_opening(sea, disk(SEA_OPEN_RADIUS))

    if SEA_CLOSE_RADIUS > 0:
        sea = binary_closing(sea, disk(SEA_CLOSE_RADIUS))

    sea = remove_small_objects(sea, min_size=SEA_MIN_AREA)
    sea, _, sea_bl = _get_border_connected(sea)

    if len(sea_bl) == 0:
        sea = border_water.copy()

    sea = _keep_largest(sea)

    if SEA_DILATE_RADIUS > 0:
        sea = binary_dilation(sea, disk(SEA_DILATE_RADIUS))

    sea = sea & candidate
    sea = remove_small_holes(sea, area_threshold=400)
    sea[~valid_mask] = False

    return sea


def _generalize_sea_mask(open_sea, valid_mask):
    sea = open_sea.copy().astype(bool)
    sea[~valid_mask] = False

    if ESTUARY_CLOSE_RADIUS > 0:
        sea = binary_closing(sea, disk(ESTUARY_CLOSE_RADIUS))

    sea = remove_small_holes(sea, area_threshold=ESTUARY_HOLE_AREA)

    if ESTUARY_OPEN_RADIUS > 0:
        sea = binary_opening(sea, disk(ESTUARY_OPEN_RADIUS))

    sea, _, bl = _get_border_connected(sea)

    if len(bl) == 0:
        sea = open_sea.copy()

    sea = _keep_largest(sea)
    sea[~valid_mask] = False

    return sea


def _smooth_linestring(line, refinements=6, window=9, simplify_tol=2.0):
    coords = np.asarray(line.coords)

    if len(coords) < 3:
        return line

    closed = np.allclose(coords[0], coords[-1])
    pts = coords[:-1] if closed else coords.copy()

    for _ in range(refinements):
        if len(pts) < 2:
            break

        if closed:
            new_pts = []
            n = len(pts)
            for i in range(n):
                p, q = pts[i], pts[(i + 1) % n]
                new_pts.extend([
                    0.75 * p + 0.25 * q,
                    0.25 * p + 0.75 * q,
                ])
            pts = np.array(new_pts)
        else:
            new_pts = [pts[0]]
            for i in range(len(pts) - 1):
                p, q = pts[i], pts[i + 1]
                new_pts.extend([
                    0.75 * p + 0.25 * q,
                    0.25 * p + 0.75 * q,
                ])
            new_pts.append(pts[-1])
            pts = np.array(new_pts)

    if window >= 3 and len(pts) >= window:
        pad = window // 2
        kernel = np.ones(window, dtype=np.float32) / window
        mode = "wrap" if closed else "edge"

        x_s = np.convolve(
            np.pad(pts[:, 0], (pad, pad), mode=mode),
            kernel,
            mode="valid",
        )
        y_s = np.convolve(
            np.pad(pts[:, 1], (pad, pad), mode=mode),
            kernel,
            mode="valid",
        )
        pts = np.column_stack([x_s, y_s])

    if closed:
        pts = np.vstack([pts, pts[0]])

    smoothed = LineString(pts)

    if simplify_tol > 0:
        smoothed = smoothed.simplify(simplify_tol, preserve_topology=False)

    return smoothed


def _extract_shorelines(sea_mask, valid_mask, transform, crs, source_name):
    sea_float = sea_mask.astype(np.float32)
    sea_float[~valid_mask] = 0.0

    sea_smooth = gaussian(
        sea_float,
        sigma=GAUSSIAN_SIGMA,
        preserve_range=True,
    )

    try:
        contours = measure.find_contours(sea_smooth, level=0.5, mask=valid_mask)
    except TypeError:
        sea_smooth[~valid_mask] = np.nan
        contours = measure.find_contours(sea_smooth, level=0.5)

    records = []

    for contour in contours:
        if contour.shape[0] < MIN_VERTICES:
            continue

        xs, ys = rasterio.transform.xy(
            transform,
            contour[:, 0],
            contour[:, 1],
            offset="center",
        )
        coords = np.column_stack([xs, ys])

        diffs = np.sqrt(np.sum(np.diff(coords, axis=0) ** 2, axis=1))
        keep = np.r_[True, diffs > 0]
        coords = coords[keep]

        if len(coords) < 2:
            continue

        line = LineString(coords)

        if line.is_empty or not line.is_valid or line.length < MIN_LINE_LENGTH_M:
            continue

        line_smooth = _smooth_linestring(
            line,
            refinements=CHAIKIN_ITERS,
            window=LINE_SMOOTH_WINDOW,
            simplify_tol=SIMPLIFY_TOL_M,
        )

        if (
            line_smooth.is_empty
            or not line_smooth.is_valid
            or line_smooth.length < MIN_LINE_LENGTH_M
        ):
            continue

        records.append({
            "source_tif": os.path.basename(source_name),
            "geometry": line_smooth,
        })

    if not records:
        return gpd.GeoDataFrame(
            columns=["source_tif", "geometry"],
            geometry="geometry",
            crs=crs,
        )

    return gpd.GeoDataFrame(records, geometry="geometry", crs=crs)