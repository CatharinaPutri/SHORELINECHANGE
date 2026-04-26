"""
==========================================================
  SCA-style Shoreline Change Analysis Module
  Computes SCE, NSM, EPR, LRR per transect

  Notes
  -----
  - Input shoreline boleh geographic (degrees) atau projected.
  - Untuk analisis, semua shoreline akan diseragamkan ke CRS meter.
  - Output web GeoJSON dikembalikan ke EPSG:4326 agar aman di Leaflet.
  - Panjang transek dihitung otomatis dari sebaran shoreline.
==========================================================
"""

from __future__ import annotations

import json
import math
import os

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import (
    GeometryCollection,
    LineString,
    MultiLineString,
    MultiPoint,
    Point,
)
from shapely.ops import linemerge, unary_union

DEFAULT_PARAMS = {
    "reference_year": 2021,
    "baseline_offset": 300,
    "baseline_placement": "land",
    "baseline_orientation": "left",
    "min_baseline_length": 200,
    "transect_spacing": 50,
    "transect_length": "auto",   # now automatic by default
    "choose_by": "distance",
    "distance_choice": "closest",
    "require_all_years": True,
    "highest_unc": 5.0,
    "epr_unc": 1.0,
}


def run_sca_analysis(
    shoreline_paths: dict,
    output_dir: str,
    params: dict | None = None,
) -> dict:
    p = {**DEFAULT_PARAMS, **(params or {})}
    os.makedirs(output_dir, exist_ok=True)

    if not shoreline_paths or len(shoreline_paths) < 2:
        raise ValueError("SCA analysis requires at least two shoreline datasets.")

    years = sorted([int(y) for y in shoreline_paths.keys()])
    ref_year = int(p["reference_year"])
    if ref_year not in years:
        ref_year = years[-1]

    raw_shore_data = {}
    for y in years:
        path = shoreline_paths[str(y)] if str(y) in shoreline_paths else shoreline_paths[y]
        raw_shore_data[y] = _load_shoreline_raw(path, year=y)

    # Prepare all shorelines in projected metric CRS
    shore_data, analysis_crs = _prepare_metric_shorelines(raw_shore_data, ref_year)

    ref_segments = shore_data[ref_year]["segments"]
    if not ref_segments:
        raise ValueError(f"No shoreline segments found for reference year {ref_year}.")

    baselines = _build_baselines(
        ref_segments,
        offset=float(p["baseline_offset"]),
        placement=p["baseline_placement"],
        orientation=p["baseline_orientation"],
        min_length=float(p["min_baseline_length"]),
    )
    if not baselines:
        raise ValueError("No baseline segments could be generated from the reference shoreline.")

    transect_length = p.get("transect_length", "auto")
    if transect_length in (None, "", 0, "auto"):
        transect_length = _estimate_transect_length(
            shore_data=shore_data,
            ref_segments=ref_segments,
            baseline_offset=float(p["baseline_offset"]),
            transect_spacing=float(p["transect_spacing"]),
        )
    else:
        transect_length = float(transect_length)

    baseline_gdf = gpd.GeoDataFrame(baselines, geometry="geometry", crs=analysis_crs)

    transects_metric_gdf = _cast_transects(
        baseline_gdf,
        spacing=float(p["transect_spacing"]),
        length=float(transect_length),
        placement=p["baseline_placement"],
        orientation=p["baseline_orientation"],
    )
    if transects_metric_gdf.empty:
        raise ValueError("No transects were generated. Check spacing/baseline settings.")

    intersections_df, _, diagnostics_df = _intersect_transects(
        transects_metric_gdf,
        shore_data,
        placement=p["baseline_placement"],
        distance_choice=p["distance_choice"],
        require_all=bool(p["require_all_years"]),
    )

    stats_df = _compute_stats(
        intersections_df,
        highest_unc=float(p["highest_unc"]),
        epr_unc=float(p["epr_unc"]),
    )

    # -------------------------------------------------------
    # Save metric outputs for analysis
    # -------------------------------------------------------
    transects_metric_path = os.path.join(output_dir, "sca_transects_metric.gpkg")
    transects_metric_gdf.to_file(transects_metric_path, layer="transects", driver="GPKG")

    diagnostics_csv_path = os.path.join(output_dir, "sca_intersection_diagnostics.csv")
    diagnostics_df.to_csv(diagnostics_csv_path, index=False)

    stats_csv_path = os.path.join(output_dir, "sca_stats.csv")
    stats_df.to_csv(stats_csv_path, index=False)

    stats_metric_path = os.path.join(output_dir, "sca_stats_metric.gpkg")
    if not stats_df.empty:
        stats_metric_geo = transects_metric_gdf[["transect_id", "geometry"]].merge(
            stats_df, on="transect_id", how="inner"
        )
        stats_metric_gdf = gpd.GeoDataFrame(
            stats_metric_geo,
            geometry="geometry",
            crs=analysis_crs,
        )
        stats_metric_gdf.to_file(stats_metric_path, layer="sca_stats", driver="GPKG")
    else:
        stats_metric_gdf = gpd.GeoDataFrame(
            columns=["transect_id", "geometry"],
            geometry="geometry",
            crs=analysis_crs,
        )

    # -------------------------------------------------------
    # Save web outputs in EPSG:4326
    # -------------------------------------------------------
    transects_web_path = os.path.join(output_dir, "sca_transects.geojson")
    transects_web_gdf = transects_metric_gdf.to_crs(epsg=4326)
    transects_web_gdf.to_file(transects_web_path, driver="GeoJSON")

    stats_geojson_path = os.path.join(output_dir, "sca_stats.geojson")
    if not stats_metric_gdf.empty:
        stats_web_gdf = stats_metric_gdf.to_crs(epsg=4326)
        stats_web_gdf.to_file(stats_geojson_path, driver="GeoJSON")
    else:
        with open(stats_geojson_path, "w", encoding="utf-8") as f:
            json.dump({"type": "FeatureCollection", "features": []}, f)

    with open(transects_web_path, "r", encoding="utf-8") as f:
        transects_geojson = json.load(f)

    with open(stats_geojson_path, "r", encoding="utf-8") as f:
        stats_geojson = json.load(f)

    summary = {}
    if not stats_df.empty:
        summary = {
            "reference_year_used": ref_year,
            "analysis_crs": str(analysis_crs),
            "web_crs": "EPSG:4326",
            "auto_transect_length_m": round(float(transect_length), 3),
            "total_transects": int(len(stats_df)),
            "mean_NSM": round(float(stats_df["NSM"].mean()), 3),
            "mean_EPR": round(float(stats_df["EPR"].mean()), 3),
            "mean_SCE": round(float(stats_df["SCE"].mean()), 3),
            "mean_LRR": round(float(stats_df["LRR"].mean()), 3),
            "accretion_count": int((stats_df["NSM_trend"] == "Akresi").sum()),
            "erosion_count": int((stats_df["NSM_trend"] == "Abrasi").sum()),
            "stable_count": int((stats_df["NSM_trend"] == "Stabil").sum()),
        }

    return {
        "transects_geojson": transects_geojson,
        "stats_geojson": stats_geojson,
        "stats_csv_path": stats_csv_path,
        "sca_csv_path": stats_csv_path,
        "transects_metric_path": transects_metric_path,
        "stats_metric_path": stats_metric_path,
        "summary": summary,
    }


def _load_shoreline_raw(path, year=None):
    gdf = gpd.read_file(path)

    if gdf.empty:
        raise ValueError(f"Empty shoreline file: {path}")

    if gdf.crs is None:
        raise ValueError(f"Shoreline CRS is missing: {path}")

    return {
        "year": year,
        "gdf": gdf,
        "crs": gdf.crs,
    }


def _prepare_metric_shorelines(raw_shore_data: dict, ref_year: int):
    crs_values = [v["gdf"].crs for v in raw_shore_data.values()]
    unique_crs = {str(crs) for crs in crs_values}

    ref_info = raw_shore_data.get(ref_year)
    if ref_info is None:
        ref_info = next(iter(raw_shore_data.values()))

    ref_gdf = ref_info["gdf"]
    ref_crs = ref_gdf.crs

    # ideal case: all same and already projected
    if len(unique_crs) == 1 and hasattr(ref_crs, "is_geographic") and not ref_crs.is_geographic:
        target_crs = ref_crs
    else:
        # if reference already projected, use it
        if hasattr(ref_crs, "is_geographic") and not ref_crs.is_geographic:
            target_crs = ref_crs
        else:
            # if reference is geographic, estimate local UTM
            ref_wgs84 = ref_gdf.to_crs(epsg=4326)
            target_crs = ref_wgs84.estimate_utm_crs()
            if target_crs is None:
                raise ValueError(
                    "Unable to estimate a projected CRS in meters from shoreline extent."
                )

    prepared = {}
    for year, info in raw_shore_data.items():
        gdf = info["gdf"]

        if str(gdf.crs) != str(target_crs):
            metric_gdf = gdf.to_crs(target_crs)
        else:
            metric_gdf = gdf.copy()

        segments = _segments_from_gdf(metric_gdf)

        prepared[year] = {
            "year": year,
            "gdf": metric_gdf,
            "segments": segments,
            "crs": metric_gdf.crs,
        }

    return prepared, target_crs


def _estimate_transect_length(
    shore_data: dict,
    ref_segments: list,
    baseline_offset: float = 300.0,
    transect_spacing: float = 50.0,
) -> float:
    """
    Estimate transect length automatically in meters.

    Logic:
    - sample all shoreline years
    - measure farthest distance to reference shoreline
    - add safety margin from baseline offset
    - round up to nearest 50 m
    """
    if not ref_segments:
        return max(1000.0, baseline_offset * 4.0)

    ref_union = unary_union(ref_segments)
    max_cross_shore_distance = 0.0
    spacing_ref = max(float(transect_spacing), 25.0)

    for info in shore_data.values():
        for seg in info["segments"]:
            if seg is None or seg.is_empty or seg.length <= 0:
                continue

            sample_count = max(8, min(120, int(seg.length / spacing_ref) + 1))
            sample_distances = np.linspace(0.0, float(seg.length), sample_count)

            for d in sample_distances:
                pt = seg.interpolate(float(d))
                dist = ref_union.distance(pt)
                if np.isfinite(dist):
                    max_cross_shore_distance = max(max_cross_shore_distance, float(dist))

    estimated = max(
        baseline_offset * 4.0,
        transect_spacing * 8.0,
        (2.0 * max_cross_shore_distance) + (2.0 * baseline_offset),
        500.0,
    )

    estimated = float(int(math.ceil(estimated / 50.0) * 50))
    return estimated


def _extract_lines(geom):
    out = []

    if geom is None or geom.is_empty:
        return out

    if isinstance(geom, LineString):
        out.append(geom)
    elif isinstance(geom, MultiLineString):
        out.extend([g for g in geom.geoms if g and not g.is_empty])
    elif isinstance(geom, GeometryCollection):
        for g in geom.geoms:
            out.extend(_extract_lines(g))

    return out


def _orient_west_to_east(line):
    coords = list(line.coords)

    if len(coords) < 2:
        return line

    if coords[0][0] > coords[-1][0]:
        return LineString(coords[::-1])

    if coords[0][0] == coords[-1][0] and coords[0][1] > coords[-1][1]:
        return LineString(coords[::-1])

    return line


def _segments_from_gdf(gdf):
    raw = []
    for geom in gdf.geometry:
        raw.extend(_extract_lines(geom))

    if not raw:
        return []

    merged = unary_union(raw)
    merged_lines = linemerge(merged) if isinstance(merged, MultiLineString) else merged

    if isinstance(merged_lines, LineString):
        segments = [merged_lines]
    elif isinstance(merged_lines, MultiLineString):
        segments = list(merged_lines.geoms)
    else:
        segments = _extract_lines(merged_lines)

    return [_orient_west_to_east(seg) for seg in segments if seg.length > 0]


def _build_baselines(ref_segments, offset=300, placement="land", orientation="left", min_length=200):
    side = (
        "left"
        if (placement == "land" and orientation == "left")
        or (placement == "sea" and orientation == "right")
        else "right"
    )

    baselines = []
    bid = 1

    for rid, seg in enumerate(ref_segments, start=1):
        try:
            off = seg.parallel_offset(offset, side=side, join_style=2)
            parts = [p for p in _extract_lines(off) if p.length >= min_length]

            for part in parts:
                baselines.append(
                    {
                        "baseline_id": bid,
                        "reference_segment_id": rid,
                        "geometry": part,
                    }
                )
                bid += 1
        except Exception as exc:
            print(f"Baseline offset failed for segment {rid}: {exc}")

    return baselines


def _cast_transects(baseline_gdf, spacing=50, length=2000, placement="land", orientation="left"):
    records = []
    tid = 1

    for _, row in baseline_gdf.iterrows():
        line = row.geometry
        if line is None or line.is_empty:
            continue

        distances = np.arange(0, line.length + spacing, spacing)
        for d in distances:
            d = min(float(d), float(line.length))
            origin = line.interpolate(d)

            eps = min(1.0, line.length * 0.001)
            p1 = line.interpolate(max(0.0, d - eps))
            p2 = line.interpolate(min(float(line.length), d + eps))
            angle = math.atan2(p2.y - p1.y, p2.x - p1.x)

            if placement == "sea":
                rot = math.radians(90) if orientation == "right" else math.radians(-90)
            else:
                rot = math.radians(-90) if orientation == "right" else math.radians(90)

            normal = angle + rot
            ux, uy = math.cos(normal), math.sin(normal)
            half = length / 2.0

            tr = LineString(
                [
                    Point(origin.x - half * ux, origin.y - half * uy),
                    Point(origin.x + half * ux, origin.y + half * uy),
                ]
            )

            records.append(
                {
                    "transect_id": tid,
                    "baseline_id": row["baseline_id"],
                    "reference_segment_id": row["reference_segment_id"],
                    "dist_on_baseline": float(d),
                    "origin_x": float(origin.x),
                    "origin_y": float(origin.y),
                    "ux": float(ux),
                    "uy": float(uy),
                    "geometry": tr,
                }
            )
            tid += 1

    return gpd.GeoDataFrame(records, geometry="geometry", crs=baseline_gdf.crs)


def _collect_points(geom):
    pts = []

    if geom is None or geom.is_empty:
        return pts

    if isinstance(geom, Point):
        pts.append(geom)
    elif isinstance(geom, MultiPoint):
        pts.extend(list(geom.geoms))
    elif isinstance(geom, LineString):
        coords = list(geom.coords)
        if coords:
            pts.append(Point(coords[0]))
            if len(coords) > 1:
                pts.append(Point(coords[-1]))
    elif isinstance(geom, (MultiLineString, GeometryCollection)):
        for g in geom.geoms:
            pts.extend(_collect_points(g))

    return pts


def _intersect_transects(
    transects_gdf,
    shore_data,
    placement="land",
    distance_choice="closest",
    require_all=True,
):
    years = sorted(shore_data.keys())
    selected_rows = []
    diagnostic_rows = []

    seaward_sign = 1 if placement == "land" else -1

    for _, tr in transects_gdf.iterrows():
        tr_line = tr.geometry
        tid = tr["transect_id"]
        ox, oy = float(tr["origin_x"]), float(tr["origin_y"])
        ux, uy = float(tr["ux"]), float(tr["uy"])

        year_hits = {}
        temp_selected = []

        for year in years:
            candidates = []

            for seg_id, seg in enumerate(shore_data[year]["segments"], start=1):
                inter = tr_line.intersection(seg)
                pts = _collect_points(inter)

                for pt in pts:
                    signed_distance = (pt.x - ox) * ux + (pt.y - oy) * uy
                    candidates.append(
                        {
                            "transect_id": tid,
                            "year": year,
                            "shore_segment_id": seg_id,
                            "signed_distance_m": float(signed_distance),
                            "abs_distance_m": float(abs(signed_distance)),
                            "point_x": float(pt.x),
                            "point_y": float(pt.y),
                        }
                    )

            dedup = []
            seen = set()
            for c in candidates:
                key = (round(c["point_x"], 3), round(c["point_y"], 3))
                if key not in seen:
                    seen.add(key)
                    dedup.append(c)

            year_hits[year] = len(dedup)
            diagnostic_rows.append(
                {
                    "transect_id": tid,
                    "year": year,
                    "n_candidates": len(dedup),
                }
            )

            if not dedup:
                continue

            seaward = [c for c in dedup if c["signed_distance_m"] * seaward_sign >= 0]
            pool = seaward if seaward else dedup

            if distance_choice == "farthest":
                best = max(pool, key=lambda x: abs(x["signed_distance_m"]))
            else:
                best = min(pool, key=lambda x: abs(x["signed_distance_m"]))

            temp_selected.append(
                {
                    "transect_id": tid,
                    "baseline_id": tr["baseline_id"],
                    "reference_segment_id": tr["reference_segment_id"],
                    "dist_on_baseline": tr["dist_on_baseline"],
                    "year": year,
                    "shore_segment_id": best["shore_segment_id"],
                    "distance_m": best["signed_distance_m"],
                    "abs_distance_m": best["abs_distance_m"],
                    "point_x": best["point_x"],
                    "point_y": best["point_y"],
                }
            )

        keep = (
            all(year_hits.get(y, 0) > 0 for y in years)
            if require_all
            else any(year_hits.get(y, 0) > 0 for y in years)
        )

        if keep:
            selected_rows.extend(temp_selected)

    intersections_df = pd.DataFrame(selected_rows)
    if not intersections_df.empty:
        intersections_df = intersections_df.sort_values(["transect_id", "year"]).reset_index(drop=True)

    diagnostics_df = pd.DataFrame(diagnostic_rows)
    return intersections_df, gpd.GeoDataFrame(), diagnostics_df


def _compute_stats(intersections_df, highest_unc=5.0, epr_unc=1.0):
    if intersections_df.empty:
        return pd.DataFrame(
            columns=[
                "transect_id",
                "baseline_id",
                "reference_segment_id",
                "dist_on_baseline",
                "oldest_year",
                "newest_year",
                "n_years",
                "SCE",
                "SCE_trend",
                "NSM",
                "NSM_trend",
                "EPR",
                "EPR_trend",
                "LRR",
            ]
        )

    def trend(value, unc):
        if -unc <= value <= unc:
            return "Stabil"
        return "Akresi" if value > unc else "Abrasi"

    rows = []
    for tid, grp in intersections_df.groupby("transect_id"):
        grp = grp.sort_values("year")
        years = grp["year"].astype(int).tolist()
        dists = grp["distance_m"].astype(float).tolist()

        oldest = min(years)
        newest = max(years)

        d_old = float(grp.loc[grp["year"] == oldest, "distance_m"].iloc[0])
        d_new = float(grp.loc[grp["year"] == newest, "distance_m"].iloc[0])

        sce = abs(float(max(dists) - min(dists)))
        nsm = float(d_new - d_old)
        dt = float(newest - oldest)
        epr = nsm / dt if dt > 0 else 0.0

        ya = np.array(years, dtype=float)
        da = np.array(dists, dtype=float)
        lrr = float(np.polyfit(ya, da, 1)[0]) if len(ya) >= 2 else 0.0

        rows.append(
            {
                "transect_id": int(tid),
                "baseline_id": int(grp["baseline_id"].iloc[0]),
                "reference_segment_id": int(grp["reference_segment_id"].iloc[0]),
                "dist_on_baseline": float(grp["dist_on_baseline"].iloc[0]),
                "oldest_year": oldest,
                "newest_year": newest,
                "n_years": int(len(years)),
                "SCE": round(sce, 3),
                "SCE_trend": trend(sce, highest_unc),
                "NSM": round(nsm, 3),
                "NSM_trend": trend(nsm, highest_unc),
                "EPR": round(epr, 3),
                "EPR_trend": trend(epr, epr_unc),
                "LRR": round(lrr, 3),
            }
        )

    return pd.DataFrame(rows).sort_values("transect_id").reset_index(drop=True)