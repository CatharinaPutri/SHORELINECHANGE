"""
==========================================================
  Shoreline Change Analysis Web Application
  Flask Backend Server
  Direct GEE -> Local GeoTIFF -> RF Shoreline Extraction
  SCA (Shoreline Change Analysis) Enabled
==========================================================
"""

from __future__ import annotations

import json
import shutil
import traceback
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

from flask import Flask, jsonify, render_template, request, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename

try:
    from .gee_dataset import (
        initialize_gee,
        generate_composite,
        export_composite_to_drive,
        get_composite_tile_url,
        download_composite_to_local,
    )
    from .rf_inference import run_shoreline_inference
    from .sca_analysis import run_sca_analysis
except ImportError:
    from gee_dataset import (
        initialize_gee,
        generate_composite,
        export_composite_to_drive,
        get_composite_tile_url,
        download_composite_to_local,
    )
    from rf_inference import run_shoreline_inference
    from sca_analysis import run_sca_analysis


BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BACKEND_DIR.parent
TEMPLATE_DIR = PROJECT_DIR / "templates"
STATIC_DIR = PROJECT_DIR / "static"
UPLOAD_DIR = BACKEND_DIR / "uploads"
OUTPUT_DIR = BACKEND_DIR / "outputs"

ALLOWED_RASTER_EXTENSIONS = {".tif", ".tiff"}

app = Flask(
    __name__,
    template_folder=str(TEMPLATE_DIR),
    static_folder=str(STATIC_DIR),
)
CORS(app)

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500 MB


def _json_ok(**payload: Any):
    payload["status"] = "ok"
    return jsonify(payload)


def _json_error(message: str, code: int = 400, **extra: Any):
    payload = {"status": "error", "message": message}
    payload.update(extra)
    return jsonify(payload), code


def _require_json() -> Dict[str, Any]:
    data = request.get_json(silent=True)
    if data is None:
        raise ValueError("Request body must be valid JSON.")
    return data


def _require_fields(data: Dict[str, Any], *fields: str) -> Tuple[Any, ...]:
    values = []
    for field in fields:
        if field not in data or data[field] in ("", None):
            raise ValueError(f"Field '{field}' is required.")
        values.append(data[field])
    return tuple(values)


def _relative_to_backend(path: Path) -> str:
    return str(path.resolve().relative_to(BACKEND_DIR.resolve())).replace("\\", "/")


def _safe_backend_path(relative_path: str) -> Path:
    candidate = (BACKEND_DIR / relative_path).resolve()
    backend_root = BACKEND_DIR.resolve()
    if backend_root not in candidate.parents and candidate != backend_root:
        raise ValueError("Invalid file path.")
    return candidate


def _prune_empty_parents(start_dir: Path, stop_at: Path) -> None:
    """
    Remove empty directories upward until stop_at.
    stop_at itself will not be removed.
    """
    current = start_dir.resolve()
    stop_at = stop_at.resolve()

    while current != stop_at and stop_at in current.parents:
        try:
            current.rmdir()
        except OSError:
            break
        current = current.parent


def _delete_file_if_exists(relative_path: str | None) -> bool:
    if not relative_path:
        return False

    path = _safe_backend_path(str(relative_path))
    if path.exists() and path.is_file():
        path.unlink()
        return True
    return False


def _delete_dir_if_exists(relative_path: str | None) -> bool:
    if not relative_path:
        return False

    path = _safe_backend_path(str(relative_path))
    if path.exists() and path.is_dir():
        shutil.rmtree(path)
        return True
    return False


def _cleanup_shoreline_item(
    output_dir_rel: str | None = None,
    source_tif_rel: str | None = None,
    geojson_path_rel: str | None = None,
) -> dict:
    """
    Delete one shoreline dataset output:
    - whole shoreline output directory
    - source tif
    - optional fallback delete by geojson path if output_dir absent
    """
    deleted_dirs: list[str] = []
    deleted_files: list[str] = []

    # 1) delete shoreline output directory if provided
    if output_dir_rel:
        output_dir = _safe_backend_path(output_dir_rel)
        if output_dir.exists() and output_dir.is_dir():
            shutil.rmtree(output_dir)
            deleted_dirs.append(output_dir_rel)
            if output_dir.parent.exists():
                _prune_empty_parents(output_dir.parent, OUTPUT_DIR)

    # 2) fallback: if no output_dir, try delete geojson + matching gpkg
    elif geojson_path_rel:
        geojson_path = _safe_backend_path(geojson_path_rel)
        if geojson_path.exists() and geojson_path.is_file():
            geojson_path.unlink()
            deleted_files.append(geojson_path_rel)

            gpkg_candidate = geojson_path.with_suffix(".gpkg")
            if gpkg_candidate.exists() and gpkg_candidate.is_file():
                gpkg_rel = _relative_to_backend(gpkg_candidate)
                gpkg_candidate.unlink()
                deleted_files.append(gpkg_rel)

            if geojson_path.parent.exists():
                _prune_empty_parents(geojson_path.parent, OUTPUT_DIR)

    # 3) delete uploaded source tif
    if source_tif_rel:
        source_tif = _safe_backend_path(source_tif_rel)
        if source_tif.exists() and source_tif.is_file():
            source_tif.unlink()
            deleted_files.append(source_tif_rel)

            if source_tif.parent.exists():
                _prune_empty_parents(source_tif.parent, UPLOAD_DIR)

    return {
        "deleted_dirs": deleted_dirs,
        "deleted_files": deleted_files,
    }


def _cleanup_sca_output(output_dir_rel: str | None = None) -> dict:
    deleted_dirs: list[str] = []

    if output_dir_rel:
        output_dir = _safe_backend_path(output_dir_rel)
        if output_dir.exists() and output_dir.is_dir():
            shutil.rmtree(output_dir)
            deleted_dirs.append(output_dir_rel)

            if output_dir.parent.exists():
                _prune_empty_parents(output_dir.parent, OUTPUT_DIR)

    return {"deleted_dirs": deleted_dirs}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/health", methods=["GET"])
def api_health():
    return _json_ok(
        message="Backend is running.",
        upload_dir=str(UPLOAD_DIR),
        output_dir=str(OUTPUT_DIR),
    )


@app.route("/api/gee/init", methods=["POST"])
def api_gee_init():
    try:
        data = _require_json()
        project_id = str(data.get("project_id", "")).strip()
        result = initialize_gee(project_id)
        return _json_ok(message=result)
    except Exception as exc:
        traceback.print_exc()
        return _json_error(str(exc), 500)


@app.route("/api/gee/composite", methods=["POST"])
def api_gee_composite():
    try:
        data = _require_json()
        roi_geojson, date_start, date_end = _require_fields(
            data, "roi", "date_start", "date_end"
        )
        cloud_max = int(data.get("cloud_max", 15))
        year_label = str(data.get("year_label", "composite"))

        composite_info = generate_composite(
            roi_geojson=roi_geojson,
            date_start=str(date_start),
            date_end=str(date_end),
            cloud_max=cloud_max,
        )

        tile_url = get_composite_tile_url(composite_info["composite_image"])

        return _json_ok(
            tile_url=tile_url,
            image_count=composite_info["image_count"],
            bands=composite_info["bands"],
            year_label=year_label,
        )
    except Exception as exc:
        traceback.print_exc()
        return _json_error(str(exc), 500)


@app.route("/api/gee/export", methods=["POST"])
def api_gee_export():
    try:
        data = _require_json()
        roi_geojson, date_start, date_end = _require_fields(
            data, "roi", "date_start", "date_end"
        )
        cloud_max = int(data.get("cloud_max", 15))
        folder_name = str(data.get("folder", "GEE_Shoreline_Export"))
        file_prefix = str(data.get("file_prefix", "S2_Composite"))
        export_scale = int(data.get("scale", 10))

        composite_info = generate_composite(
            roi_geojson=roi_geojson,
            date_start=str(date_start),
            date_end=str(date_end),
            cloud_max=cloud_max,
        )

        task_id = export_composite_to_drive(
            composite_image=composite_info["composite_image"],
            roi_geojson=roi_geojson,
            folder=folder_name,
            file_prefix=file_prefix,
            scale=export_scale,
        )

        return _json_ok(
            message=f"Export task started: {task_id}",
            task_id=task_id,
        )
    except Exception as exc:
        traceback.print_exc()
        return _json_error(str(exc), 500)


@app.route("/api/shoreline/extract", methods=["POST"])
def api_shoreline_extract():
    """
    Manual fallback:
    Upload a local GeoTIFF, then run RF shoreline extraction.
    Model path is handled internally inside rf_inference.py.
    """
    try:
        if "tif_file" not in request.files:
            return _json_error("No TIF file uploaded.", 400)

        tif_file = request.files["tif_file"]
        year_label = str(request.form.get("year_label", "unknown")).strip() or "unknown"

        original_name = secure_filename(tif_file.filename or "")
        if not original_name:
            return _json_error("Uploaded file name is invalid.", 400)

        ext = Path(original_name).suffix.lower()
        if ext not in ALLOWED_RASTER_EXTENSIONS:
            return _json_error("Only .tif or .tiff files are allowed.", 400)

        session_id = str(uuid.uuid4())[:8]
        session_dir = UPLOAD_DIR / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        tif_path = session_dir / original_name
        tif_file.save(str(tif_path))

        out_dir = OUTPUT_DIR / f"shoreline_{year_label}_{session_id}"
        out_dir.mkdir(parents=True, exist_ok=True)

        geojson_path = Path(
            run_shoreline_inference(
                tif_path=str(tif_path),
                output_dir=str(out_dir),
                year_label=year_label,
            )
        )

        with geojson_path.open("r", encoding="utf-8") as f:
            shoreline_geojson = json.load(f)

        return _json_ok(
            year_label=year_label,
            session_id=session_id,
            geojson=shoreline_geojson,
            geojson_path=_relative_to_backend(geojson_path),
            output_dir=_relative_to_backend(out_dir),
            source_tif=_relative_to_backend(tif_path),
            mode="manual_upload",
        )
    except Exception as exc:
        traceback.print_exc()
        return _json_error(str(exc), 500, traceback=traceback.format_exc())


@app.route("/api/shoreline/extract-from-gee", methods=["POST"])
def api_shoreline_extract_from_gee():
    """
    Direct workflow:
    GEE composite -> download to local GeoTIFF -> RF shoreline extraction
    """
    try:
        data = _require_json()
        roi_geojson, date_start, date_end = _require_fields(
            data, "roi", "date_start", "date_end"
        )

        year_label = str(data.get("year_label", "unknown")).strip() or "unknown"
        cloud_max = int(data.get("cloud_max", 15))
        scale = int(data.get("scale", 10))
        crs = str(data.get("crs", "EPSG:32749"))

        session_id = str(uuid.uuid4())[:8]
        session_dir = UPLOAD_DIR / f"gee_{session_id}"
        session_dir.mkdir(parents=True, exist_ok=True)

        tif_path = session_dir / f"S2_{year_label}_Composite_7Band.tif"

        download_info = download_composite_to_local(
            roi_geojson=roi_geojson,
            date_start=str(date_start),
            date_end=str(date_end),
            out_tif_path=str(tif_path),
            cloud_max=cloud_max,
            scale=scale,
            crs=crs,
        )

        out_dir = OUTPUT_DIR / f"shoreline_{year_label}_{session_id}"
        out_dir.mkdir(parents=True, exist_ok=True)

        geojson_path = Path(
            run_shoreline_inference(
                tif_path=str(download_info["tif_path"]),
                output_dir=str(out_dir),
                year_label=year_label,
            )
        )

        with geojson_path.open("r", encoding="utf-8") as f:
            shoreline_geojson = json.load(f)

        return _json_ok(
            year_label=year_label,
            session_id=session_id,
            image_count=download_info["image_count"],
            bands=download_info["bands"],
            geojson=shoreline_geojson,
            geojson_path=_relative_to_backend(geojson_path),
            output_dir=_relative_to_backend(out_dir),
            source_tif=_relative_to_backend(Path(download_info["tif_path"])),
            mode="direct_from_gee",
        )
    except Exception as exc:
        traceback.print_exc()
        return _json_error(str(exc), 500, traceback=traceback.format_exc())


@app.route("/api/shoreline/delete", methods=["POST"])
def api_shoreline_delete():
    """
    Delete one extracted shoreline dataset.

    Expected JSON:
    {
      "year_label": "2019",
      "output_dir": "outputs/shoreline_2019_abcd1234",
      "source_tif": "uploads/abcd1234/file.tif",
      "geojson_path": "outputs/shoreline_2019_abcd1234/file_shoreline.geojson"
    }
    """
    try:
        data = _require_json()

        year_label = str(data.get("year_label", "")).strip()
        output_dir = data.get("output_dir")
        source_tif = data.get("source_tif")
        geojson_path = data.get("geojson_path")

        if not any([output_dir, source_tif, geojson_path]):
            return _json_error(
                "At least one of 'output_dir', 'source_tif', or 'geojson_path' is required.",
                400,
            )

        result = _cleanup_shoreline_item(
            output_dir_rel=str(output_dir) if output_dir else None,
            source_tif_rel=str(source_tif) if source_tif else None,
            geojson_path_rel=str(geojson_path) if geojson_path else None,
        )

        return _json_ok(
            message=f"Shoreline {year_label or ''} deleted successfully.".strip(),
            year_label=year_label or None,
            deleted_dirs=result["deleted_dirs"],
            deleted_files=result["deleted_files"],
        )
    except Exception as exc:
        traceback.print_exc()
        return _json_error(str(exc), 500, traceback=traceback.format_exc())


@app.route("/api/sca/analyze", methods=["POST"])
@app.route("/api/dsas/analyze", methods=["POST"])  # alias sementara untuk kompatibilitas
def api_sca_analyze():
    """
    Shoreline Change Analysis (SCA)

    Notes:
    - Uses shoreline files produced by shoreline extraction.
    - Metric analysis is handled inside sca_analysis.py.
    - Web output is returned in EPSG:4326 for Leaflet.
    """
    try:
        data = _require_json()
        shorelines = data.get("shorelines", {})
        params = data.get("params", {})

        if not isinstance(shorelines, dict) or len(shorelines) < 2:
            return _json_error("At least two shoreline paths are required.", 400)

        shoreline_paths = {}
        for year, path_str in shorelines.items():
            shoreline_paths[str(year)] = str(_safe_backend_path(str(path_str)))

        session_id = str(uuid.uuid4())[:8]
        out_dir = OUTPUT_DIR / f"sca_{session_id}"
        out_dir.mkdir(parents=True, exist_ok=True)

        result = run_sca_analysis(
            shoreline_paths=shoreline_paths,
            output_dir=str(out_dir),
            params=params,
        )

        stats_csv_abs = Path(result["stats_csv_path"]).resolve()

        return _json_ok(
            session_id=session_id,
            output_dir=_relative_to_backend(out_dir),
            sca_output_dir=_relative_to_backend(out_dir),
            transects_geojson=result["transects_geojson"],
            stats_geojson=result["stats_geojson"],
            stats_csv_path=_relative_to_backend(stats_csv_abs),
            sca_csv_path=_relative_to_backend(stats_csv_abs),
            summary=result["summary"],
        )
    except Exception as exc:
        traceback.print_exc()
        return _json_error(str(exc), 500, traceback=traceback.format_exc())


@app.route("/api/sca/delete", methods=["POST"])
def api_sca_delete():
    """
    Delete one SCA analysis output directory.

    Expected JSON:
    {
      "output_dir": "outputs/sca_abcd1234"
    }
    """
    try:
        data = _require_json()
        output_dir = data.get("output_dir")

        if not output_dir:
            return _json_error("Field 'output_dir' is required.", 400)

        result = _cleanup_sca_output(output_dir_rel=str(output_dir))

        return _json_ok(
            message="SCA output deleted successfully.",
            deleted_dirs=result["deleted_dirs"],
        )
    except Exception as exc:
        traceback.print_exc()
        return _json_error(str(exc), 500, traceback=traceback.format_exc())


@app.route("/api/session/reset", methods=["POST"])
def api_session_reset():
    """
    Bulk reset backend outputs for current frontend session.

    Expected JSON:
    {
      "shorelines": [
        {
          "year_label": "2018",
          "output_dir": "outputs/shoreline_2018_xxxx",
          "source_tif": "uploads/xxxx/file.tif",
          "geojson_path": "outputs/shoreline_2018_xxxx/file_shoreline.geojson"
        }
      ],
      "sca_output_dir": "outputs/sca_yyyy"
    }
    """
    try:
        data = _require_json()

        shoreline_items = data.get("shorelines", [])
        sca_output_dir = data.get("sca_output_dir")

        if shoreline_items is None:
            shoreline_items = []

        if not isinstance(shoreline_items, list):
            return _json_error("Field 'shorelines' must be a list.", 400)

        deleted_dirs: list[str] = []
        deleted_files: list[str] = []

        for item in shoreline_items:
            if not isinstance(item, dict):
                continue

            result = _cleanup_shoreline_item(
                output_dir_rel=str(item.get("output_dir")) if item.get("output_dir") else None,
                source_tif_rel=str(item.get("source_tif")) if item.get("source_tif") else None,
                geojson_path_rel=str(item.get("geojson_path")) if item.get("geojson_path") else None,
            )
            deleted_dirs.extend(result["deleted_dirs"])
            deleted_files.extend(result["deleted_files"])

        if sca_output_dir:
            sca_result = _cleanup_sca_output(output_dir_rel=str(sca_output_dir))
            deleted_dirs.extend(sca_result["deleted_dirs"])

        return _json_ok(
            message="Session outputs deleted successfully.",
            deleted_dirs=deleted_dirs,
            deleted_files=deleted_files,
        )
    except Exception as exc:
        traceback.print_exc()
        return _json_error(str(exc), 500, traceback=traceback.format_exc())


@app.route("/api/download/<path:filepath>")
def api_download(filepath):
    try:
        full_path = _safe_backend_path(filepath)
        if full_path.exists() and full_path.is_file():
            return send_file(str(full_path), as_attachment=True)
        return _json_error("File not found.", 404)
    except Exception as exc:
        return _json_error(str(exc), 400)


if __name__ == "__main__":
    print("=" * 60)
    print("  Shoreline Change Analysis Web Application")
    print("  Direct GEE -> RF Shoreline Extraction Enabled")
    print("  SCA (Shoreline Change Analysis) Enabled")
    print(f"  Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    app.run(debug=True, host="0.0.0.0", port=5000)