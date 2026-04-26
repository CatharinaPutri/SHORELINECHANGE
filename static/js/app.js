/**
 * ==========================================================
 *  Shoreline Change Analysis — Frontend Application
 *  Supports:
 *  1) GEE preview
 *  2) Export to Drive
 *  3) Direct shoreline extraction from GEE
 *  4) Manual upload fallback
 *  5) SCA analysis
 *  6) ROI hide/show in Active Layers
 *  7) Delete shoreline by year
 *  8) Clear map results
 *  9) Reset session & delete backend outputs
 * ==========================================================
 */

"use strict";

/* ==========================================================
   GLOBAL STATE
   ========================================================== */
const APP = {
    map: null,
    drawnItems: null,
    currentROI: null,
    roiVisible: true,
    geeConnected: false,

    compositeLayers: {},   // { year: TileLayer }
    shorelines: {},        // { year: { geojson, path, outputDir, sourceTif, color, layer, yearLabel } }

    transectsLayer: null,
    statsLayer: null,
    scaCSVPath: null,
    scaOutputDir: null,
};

const SHORE_COLORS = {
    "2018": "#00b4d8",
    "2019": "#0096c7",
    "2020": "#06d6a0",
    "2021": "#ffd166",
    "2022": "#ef476f",
    "2023": "#118ab2",
    "2024": "#e76f51",
};

const TREND_STYLES = {
    Akresi: { color: "#06d6a0", weight: 2.5, opacity: 0.95 },
    Abrasi: { color: "#ef476f", weight: 2.5, opacity: 0.95 },
    Stabil: { color: "#ffd166", weight: 2.5, opacity: 0.95 },
};

/* ==========================================================
   HELPERS
   ========================================================== */
const $ = (id) => document.getElementById(id);

function getShoreColor(year) {
    return SHORE_COLORS[String(year)] || "#ffffff";
}

function escapeHtml(value) {
    return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
}

function hasFeatures(fc) {
    return !!(fc && Array.isArray(fc.features) && fc.features.length > 0);
}

function formatNum(value, digits = 3) {
    const n = Number(value);
    if (!Number.isFinite(n)) return "-";
    return n.toFixed(digits);
}

async function parseJSONResponse(res) {
    let data = null;

    try {
        data = await res.json();
    } catch (_) {
        throw new Error(`Server returned ${res.status} ${res.statusText}`);
    }

    if (!res.ok || data.status === "error") {
        throw new Error(data?.message || `Request failed with ${res.status}`);
    }

    return data;
}

function setButtonBusy(button, busy, busyHTML, idleHTML) {
    if (!button) return;
    button.disabled = busy;
    button.innerHTML = busy ? busyHTML : idleHTML;
}

function showToast(type, title, message) {
    const container = $("toastContainer");
    if (!container) return;

    const icons = {
        success: "fas fa-check-circle",
        error: "fas fa-exclamation-circle",
        warning: "fas fa-exclamation-triangle",
        info: "fas fa-info-circle",
    };

    const toast = document.createElement("div");
    toast.className = `toast toast-${type}`;

    const icon = document.createElement("i");
    icon.className = `${icons[type] || icons.info} toast-icon`;

    const body = document.createElement("div");
    body.className = "toast-body";

    const titleEl = document.createElement("div");
    titleEl.className = "toast-title";
    titleEl.textContent = title;

    const messageEl = document.createElement("div");
    messageEl.className = "toast-message";
    messageEl.textContent = message;

    body.appendChild(titleEl);
    body.appendChild(messageEl);
    toast.appendChild(icon);
    toast.appendChild(body);
    container.appendChild(toast);

    setTimeout(() => {
        toast.style.animation = "slideOut 0.3s ease forwards";
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}

function showLoading(text = "Processing...") {
    $("loadingOverlay")?.classList.add("active");
    if ($("loadingText")) $("loadingText").textContent = text;
}

function hideLoading() {
    $("loadingOverlay")?.classList.remove("active");
}

function setProgress(containerId, fillId, textId, pct, text, visible = true) {
    const container = $(containerId);
    const fill = $(fillId);
    const textEl = $(textId);

    if (container) container.style.display = visible ? "block" : "none";
    if (fill) fill.style.width = `${Math.max(0, Math.min(100, pct))}%`;
    if (textEl) textEl.textContent = text;
}

function clearProgress(containerId, fillId, textId, resetText = "Waiting...") {
    const container = $(containerId);
    const fill = $(fillId);
    const textEl = $(textId);

    if (container) container.style.display = "none";
    if (fill) fill.style.width = "0%";
    if (textEl) textEl.textContent = resetText;
}

function confirmAction(message) {
    return window.confirm(message);
}

/* ==========================================================
   INITIALIZATION
   ========================================================== */
document.addEventListener("DOMContentLoaded", () => {
    initMap();
    initTabs();
    initSidebar();
    initGEE();
    initDatasetControls();
    initUploadControls();
    initShorelineExtraction();
    initSCA();
    initSessionControls();

    updateROIInfo();
    updateLayerList();
    updateLegend();
});

/* ==========================================================
   MAP
   ========================================================== */
function initMap() {
    APP.map = L.map("map", {
        center: [-7.8, 110.4],
        zoom: 11,
        zoomControl: true,
    });

    const dark = L.tileLayer(
        "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
        { attribution: '&copy; <a href="https://carto.com/">CARTO</a>', maxZoom: 19 }
    );

    const satellite = L.tileLayer(
        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        { attribution: "&copy; Esri", maxZoom: 19 }
    );

    const osm = L.tileLayer(
        "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
        { attribution: "&copy; OpenStreetMap", maxZoom: 19 }
    );

    dark.addTo(APP.map);

    L.control.layers(
        {
            "🌑 Dark": dark,
            "🛰️ Satellite": satellite,
            "🗺️ OSM": osm,
        },
        null,
        { position: "topright" }
    ).addTo(APP.map);

    APP.drawnItems = new L.FeatureGroup();
    APP.map.addLayer(APP.drawnItems);

    const drawControl = new L.Control.Draw({
        position: "topleft",
        draw: {
            polygon: {
                shapeOptions: {
                    color: "#00b4d8",
                    weight: 2,
                    fillOpacity: 0.1,
                },
            },
            rectangle: {
                shapeOptions: {
                    color: "#00b4d8",
                    weight: 2,
                    fillOpacity: 0.1,
                },
            },
            circle: false,
            circlemarker: false,
            marker: false,
            polyline: false,
        },
        edit: {
            featureGroup: APP.drawnItems,
            edit: true,
            remove: true,
        },
    });

    APP.map.addControl(drawControl);

    APP.map.on(L.Draw.Event.CREATED, (e) => {
        APP.drawnItems.clearLayers();
        APP.drawnItems.addLayer(e.layer);
        APP.currentROI = e.layer.toGeoJSON().geometry;

        APP.roiVisible = true;
        if (!APP.map.hasLayer(APP.drawnItems)) {
            APP.map.addLayer(APP.drawnItems);
        }

        updateROIInfo();
        updateLayerList();
    });

    APP.map.on(L.Draw.Event.EDITED, () => {
        const layers = APP.drawnItems.getLayers();
        APP.currentROI = layers.length ? layers[0].toGeoJSON().geometry : null;

        updateROIInfo();
        updateLayerList();
    });

    APP.map.on(L.Draw.Event.DELETED, () => {
        APP.currentROI = null;
        APP.roiVisible = true;

        if (!APP.map.hasLayer(APP.drawnItems)) {
            APP.map.addLayer(APP.drawnItems);
        }

        updateROIInfo();
        updateLayerList();
    });

    APP.map.on("mousemove", (e) => {
        if ($("mapCoords")) {
            $("mapCoords").textContent = `${e.latlng.lat.toFixed(5)}, ${e.latlng.lng.toFixed(5)}`;
        }
    });

    APP.map.on("zoomend", () => {
        if ($("mapZoom")) {
            $("mapZoom").textContent = APP.map.getZoom();
        }
    });

    if ($("mapZoom")) {
        $("mapZoom").textContent = APP.map.getZoom();
    }
}

function updateROIInfo() {
    const el = $("roiInfo");
    if (!el) return;

    if (!APP.currentROI) {
        el.innerHTML = `<span class="tag tag-muted">No ROI defined</span>`;
        return;
    }

    const coords = APP.currentROI?.coordinates?.[0] || [];
    const vertexCount = Math.max(0, coords.length - 1);

    el.innerHTML = `
        <span class="tag tag-success">
            <i class="fas fa-check"></i> ROI defined (${vertexCount} vertices)
        </span>
    `;
}

function setROIVisibility(visible) {
    APP.roiVisible = !!visible;

    const onMap = APP.map.hasLayer(APP.drawnItems);

    if (APP.roiVisible) {
        if (!onMap) APP.map.addLayer(APP.drawnItems);
    } else {
        if (onMap) APP.map.removeLayer(APP.drawnItems);
    }

    updateLayerList();
}

function toggleROIVisibility() {
    setROIVisibility(!APP.roiVisible);
}

/* ==========================================================
   TABS & SIDEBAR
   ========================================================== */
function initTabs() {
    document.querySelectorAll(".tab-btn").forEach((btn) => {
        btn.addEventListener("click", () => {
            document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
            document.querySelectorAll(".tab-content").forEach((c) => c.classList.remove("active"));

            btn.classList.add("active");
            const target = document.getElementById(btn.dataset.tab);
            if (target) target.classList.add("active");
        });
    });
}

function initSidebar() {
    const toggle = $("sidebarToggle");
    const sidebar = $("sidebar");
    if (!toggle || !sidebar) return;

    toggle.addEventListener("click", () => {
        sidebar.classList.toggle("collapsed");
        setTimeout(() => APP.map.invalidateSize(), 300);
    });
}

/* ==========================================================
   GEE AUTHENTICATION
   ========================================================== */
function initGEE() {
    const btn = $("btnGeeInit");
    if (!btn) return;

    btn.addEventListener("click", async () => {
        const projectId = $("geeProjectId")?.value?.trim() || "";
        if (!projectId) {
            showToast("warning", "Missing Project ID", "Please enter your GEE project ID.");
            return;
        }

        setButtonBusy(
            btn,
            true,
            '<i class="fas fa-spinner fa-spin"></i> Connecting...',
            '<i class="fas fa-plug"></i> Connect to GEE'
        );

        try {
            const res = await fetch("/api/gee/init", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ project_id: projectId }),
            });

            const data = await parseJSONResponse(res);

            APP.geeConnected = true;

            const status = $("geeStatus");
            if (status) {
                status.classList.add("connected");
                status.innerHTML = '<i class="fas fa-circle"></i> GEE: Connected';
            }

            showToast("success", "Connected", data.message || "Google Earth Engine connected.");
        } catch (err) {
            showToast("error", "GEE Connection Failed", err.message);
        } finally {
            setButtonBusy(
                btn,
                false,
                "",
                '<i class="fas fa-plug"></i> Connect to GEE'
            );
        }
    });
}

/* ==========================================================
   DATASET CONTROLS
   ========================================================== */
let datasetCount = 1;

function initDatasetControls() {
    $("btnAddDataset")?.addEventListener("click", addDatasetCard);
    $("btnGeneratePreview")?.addEventListener("click", generatePreview);
    $("btnExportDrive")?.addEventListener("click", exportToDrive);
}

function addDatasetCard() {
    datasetCount += 1;

    const defaults = { 2: "2021", 3: "2024" };
    const year = defaults[datasetCount] || String(2016 + datasetCount);

    const container = $("datasetEntries");
    if (!container) return;

    const card = document.createElement("div");
    card.className = "dataset-card";
    card.innerHTML = `
        <div class="dataset-header">
            <span class="dataset-badge">Dataset ${datasetCount}</span>
            <button class="btn-icon btn-remove-dataset" type="button" title="Remove">
                <i class="fas fa-times"></i>
            </button>
        </div>
        <div class="form-row">
            <div class="form-group">
                <label>Year Label</label>
                <input type="text" class="ds-year" value="${escapeHtml(year)}" />
            </div>
            <div class="form-group">
                <label>Cloud Max (%)</label>
                <input type="number" class="ds-cloud" value="15" min="0" max="100" />
            </div>
        </div>
        <div class="form-row">
            <div class="form-group">
                <label>Start Date</label>
                <input type="date" class="ds-start" value="${year}-01-01" />
            </div>
            <div class="form-group">
                <label>End Date</label>
                <input type="date" class="ds-end" value="${year}-12-31" />
            </div>
        </div>
    `;

    container.appendChild(card);
    card.querySelector(".btn-remove-dataset")?.addEventListener("click", () => card.remove());
}

function getDatasetEntries() {
    const entries = [];

    document.querySelectorAll(".dataset-card").forEach((card) => {
        const yearLabel = card.querySelector(".ds-year")?.value?.trim() || "";
        const cloudMax = parseInt(card.querySelector(".ds-cloud")?.value, 10);
        const dateStart = card.querySelector(".ds-start")?.value || "";
        const dateEnd = card.querySelector(".ds-end")?.value || "";

        if (!yearLabel || !dateStart || !dateEnd) return;

        entries.push({
            year_label: yearLabel,
            cloud_max: Number.isFinite(cloudMax) ? cloudMax : 15,
            date_start: dateStart,
            date_end: dateEnd,
        });
    });

    return entries;
}

async function generatePreview() {
    if (!APP.geeConnected) {
        showToast("warning", "Not Connected", "Connect to GEE first.");
        return;
    }

    if (!APP.currentROI) {
        showToast("warning", "No ROI", "Draw a region on the map first.");
        return;
    }

    const entries = getDatasetEntries();
    if (!entries.length) {
        showToast("warning", "No Datasets", "Add at least one valid dataset.");
        return;
    }

    showLoading("Generating composite preview...");

    try {
        for (const entry of entries) {
            const res = await fetch("/api/gee/composite", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    roi: APP.currentROI,
                    date_start: entry.date_start,
                    date_end: entry.date_end,
                    cloud_max: entry.cloud_max,
                    year_label: entry.year_label,
                }),
            });

            const data = await parseJSONResponse(res);

            if (APP.compositeLayers[entry.year_label]) {
                APP.map.removeLayer(APP.compositeLayers[entry.year_label]);
            }

            const tileLayer = L.tileLayer(data.tile_url, {
                maxZoom: 19,
                opacity: 0.9,
            });

            tileLayer.addTo(APP.map);
            APP.compositeLayers[entry.year_label] = tileLayer;

            showToast(
                "success",
                `Composite ${entry.year_label}`,
                `${data.image_count} images composited`
            );
        }

        updateLayerList();
        updateLegend();
    } catch (err) {
        showToast("error", "Preview Failed", err.message);
    } finally {
        hideLoading();
    }
}

async function exportToDrive() {
    if (!APP.geeConnected) {
        showToast("warning", "Not Connected", "Connect to GEE first.");
        return;
    }

    if (!APP.currentROI) {
        showToast("warning", "No ROI", "Draw a region on the map first.");
        return;
    }

    const entries = getDatasetEntries();
    if (!entries.length) {
        showToast("warning", "No Datasets", "Add at least one valid dataset.");
        return;
    }

    showLoading("Starting export tasks...");

    try {
        for (const entry of entries) {
            const res = await fetch("/api/gee/export", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    roi: APP.currentROI,
                    date_start: entry.date_start,
                    date_end: entry.date_end,
                    cloud_max: entry.cloud_max,
                    folder: `GEE_Shoreline_Export_${entry.year_label}`,
                    file_prefix: `S2_${entry.year_label}_Composite_7Band`,
                    scale: 10,
                }),
            });

            const data = await parseJSONResponse(res);

            showToast(
                "success",
                `Export ${entry.year_label}`,
                data.message || "Export task started."
            );
        }
    } catch (err) {
        showToast("error", "Export Failed", err.message);
    } finally {
        hideLoading();
    }
}

/* ==========================================================
   SHORELINE EXTRACTION
   ========================================================== */
function initUploadControls() {
    $("btnAddUpload")?.addEventListener("click", () => {
        const container = $("uploadEntries");
        if (!container) return;

        const card = document.createElement("div");
        card.className = "upload-card";
        card.innerHTML = `
            <div class="dataset-header">
                <span class="dataset-badge" style="font-size:10px;">Additional Year</span>
                <button class="btn-icon btn-remove-dataset" type="button" title="Remove">
                    <i class="fas fa-times"></i>
                </button>
            </div>
            <div class="form-group">
                <label>Year Label</label>
                <input type="text" class="ul-year" value="" placeholder="e.g. 2021" />
            </div>
            <div class="file-input-wrapper">
                <input type="file" class="ul-file" accept=".tif,.tiff" />
            </div>
        `;

        container.appendChild(card);
        card.querySelector(".btn-remove-dataset")?.addEventListener("click", () => card.remove());
    });
}

function initShorelineExtraction() {
    $("btnExtractShoreline")?.addEventListener("click", extractShorelines);
}

function getUploadEntries() {
    const uploads = [];

    document.querySelectorAll(".upload-card").forEach((card) => {
        const year = card.querySelector(".ul-year")?.value?.trim() || "";
        const fileInput = card.querySelector(".ul-file");

        if (year && fileInput?.files?.length > 0) {
            uploads.push({
                year,
                file: fileInput.files[0],
            });
        }
    });

    return uploads;
}

function renderShorelineResults() {
    const container = $("shorelineResultsList");
    if (!container) return;

    container.innerHTML = "";

    const years = Object.keys(APP.shorelines).sort();
    years.forEach((year) => {
        const item = APP.shorelines[year];
        const count = item?.geojson?.features?.length || 0;

        const row = document.createElement("div");
        row.className = "layer-item";

        row.innerHTML = `
            <div class="layer-color" style="background:${escapeHtml(item.color)};"></div>
            <span class="layer-name">Shoreline ${escapeHtml(year)} (${count} segments)</span>
        `;

        const actions = document.createElement("div");
        actions.style.display = "flex";
        actions.style.alignItems = "center";
        actions.style.gap = "8px";

        const okTag = document.createElement("span");
        okTag.className = "tag tag-success";
        okTag.style.fontSize = "10px";
        okTag.textContent = "✓";

        const deleteBtn = document.createElement("button");
        deleteBtn.type = "button";
        deleteBtn.className = "btn-icon";
        deleteBtn.title = `Delete shoreline ${year}`;
        deleteBtn.innerHTML = '<i class="fas fa-trash"></i>';
        deleteBtn.addEventListener("click", async () => {
            await deleteShorelineYear(year);
        });

        actions.appendChild(okTag);
        actions.appendChild(deleteBtn);
        row.appendChild(actions);
        container.appendChild(row);
    });

    if ($("shorelineResults")) {
        $("shorelineResults").style.display = years.length ? "block" : "none";
    }
}

function renderShorelineOnMap(year, data) {
    if (APP.shorelines[year]?.layer) {
        APP.map.removeLayer(APP.shorelines[year].layer);
    }

    const color = getShoreColor(year);

    const layer = L.geoJSON(data.geojson, {
        style: {
            color,
            weight: 2.5,
            opacity: 0.95,
        },
        onEachFeature: (feature, layerRef) => {
            layerRef.bindPopup(`
                <strong>Shoreline ${escapeHtml(year)}</strong><br>
                Source: ${escapeHtml(feature.properties?.source_tif || "-")}
            `);
        },
    });

    layer.addTo(APP.map);

    APP.shorelines[year] = {
        yearLabel: year,
        geojson: data.geojson,
        path: data.geojson_path,
        outputDir: data.output_dir,
        sourceTif: data.source_tif || null,
        color,
        layer,
    };

    if (layer.getBounds().isValid()) {
        APP.map.fitBounds(layer.getBounds(), { padding: [40, 40] });
    }

    const count = data.geojson?.features?.length || 0;
    return count;
}

async function extractShorelines() {
    const uploads = getUploadEntries();

    if (uploads.length > 0) {
        await extractShorelinesFromUploads(uploads);
        return;
    }

    await extractShorelinesDirectFromGEE();
}

async function extractShorelinesFromUploads(uploads) {
    setProgress(
        "shorelineProgress",
        "shorelineProgressFill",
        "shorelineProgressText",
        0,
        "Starting manual extraction...",
        true
    );
    showLoading("Running shoreline extraction from uploaded GeoTIFF...");

    try {
        for (let i = 0; i < uploads.length; i++) {
            const { year, file } = uploads[i];
            const pct = Math.round((i / uploads.length) * 100);

            setProgress(
                "shorelineProgress",
                "shorelineProgressFill",
                "shorelineProgressText",
                pct,
                `Processing upload ${year}... (${i + 1}/${uploads.length})`,
                true
            );

            const formData = new FormData();
            formData.append("tif_file", file);
            formData.append("year_label", year);

            const res = await fetch("/api/shoreline/extract", {
                method: "POST",
                body: formData,
            });

            const data = await parseJSONResponse(res);
            const count = renderShorelineOnMap(year, data);

            showToast("success", `Shoreline ${year}`, `${count} segments extracted from upload`);
        }

        setProgress(
            "shorelineProgress",
            "shorelineProgressFill",
            "shorelineProgressText",
            100,
            "Manual shoreline extraction complete!",
            true
        );

        clearSCAResultsUIOnly();
        updateSCAYearOptions();
        renderShorelineResults();
        updateLayerList();
        updateLegend();
    } catch (err) {
        showToast("error", "Manual Shoreline Extraction Failed", err.message);
    } finally {
        hideLoading();
    }
}

async function extractShorelinesDirectFromGEE() {
    if (!APP.geeConnected) {
        showToast("warning", "Not Connected", "Connect to GEE first.");
        return;
    }

    if (!APP.currentROI) {
        showToast("warning", "No ROI", "Draw a region on the map first.");
        return;
    }

    const entries = getDatasetEntries();
    if (!entries.length) {
        showToast("warning", "No Datasets", "Fill dataset parameters first in the Dataset tab.");
        return;
    }

    setProgress(
        "shorelineProgress",
        "shorelineProgressFill",
        "shorelineProgressText",
        0,
        "Starting direct extraction from GEE...",
        true
    );
    showLoading("Downloading composite from GEE and extracting shoreline...");

    try {
        for (let i = 0; i < entries.length; i++) {
            const entry = entries[i];
            const pct = Math.round((i / entries.length) * 100);

            setProgress(
                "shorelineProgress",
                "shorelineProgressFill",
                "shorelineProgressText",
                pct,
                `Processing GEE dataset ${entry.year_label}... (${i + 1}/${entries.length})`,
                true
            );

            const res = await fetch("/api/shoreline/extract-from-gee", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    roi: APP.currentROI,
                    date_start: entry.date_start,
                    date_end: entry.date_end,
                    cloud_max: entry.cloud_max,
                    year_label: entry.year_label,
                    scale: 10,
                    crs: "EPSG:32749",
                }),
            });

            const data = await parseJSONResponse(res);
            const count = renderShorelineOnMap(entry.year_label, data);

            showToast(
                "success",
                `Shoreline ${entry.year_label}`,
                `${count} segments extracted directly from GEE (${data.image_count} images)`
            );
        }

        setProgress(
            "shorelineProgress",
            "shorelineProgressFill",
            "shorelineProgressText",
            100,
            "Direct shoreline extraction complete!",
            true
        );

        clearSCAResultsUIOnly();
        updateSCAYearOptions();
        renderShorelineResults();
        updateLayerList();
        updateLegend();
    } catch (err) {
        showToast("error", "Direct Shoreline Extraction Failed", err.message);
    } finally {
        hideLoading();
    }
}

async function deleteShorelineYear(year) {
    const item = APP.shorelines[year];
    if (!item) return;

    const ok = confirmAction(
        `Delete shoreline year ${year} and its generated backend files?`
    );
    if (!ok) return;

    showLoading(`Deleting shoreline ${year}...`);

    try {
        // delete invalid SCA output first if present
        await deleteSCAOutputBackendOnly();

        const res = await fetch("/api/shoreline/delete", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                year_label: year,
                output_dir: item.outputDir,
                source_tif: item.sourceTif,
                geojson_path: item.path,
            }),
        });

        await parseJSONResponse(res);

        if (item.layer && APP.map.hasLayer(item.layer)) {
            APP.map.removeLayer(item.layer);
        }

        delete APP.shorelines[year];

        clearSCAResultsUIOnly();
        updateSCAYearOptions();
        renderShorelineResults();
        updateLayerList();
        updateLegend();

        showToast("success", "Shoreline Deleted", `Shoreline ${year} deleted successfully.`);
    } catch (err) {
        showToast("error", "Delete Failed", err.message);
    } finally {
        hideLoading();
    }
}

function updateSCAYearOptions() {
    const select = $("scaRefYear");
    if (!select) return;

    const years = Object.keys(APP.shorelines).sort();
    select.innerHTML = "";

    if (!years.length) {
        const opt = document.createElement("option");
        opt.value = "";
        opt.textContent = "Select shoreline year";
        select.appendChild(opt);
        return;
    }

    years.forEach((year) => {
        const opt = document.createElement("option");
        opt.value = year;
        opt.textContent = year;
        select.appendChild(opt);
    });
}

/* ==========================================================
   SCA ANALYSIS
   ========================================================== */
function initSCA() {
    $("btnRunSCA")?.addEventListener("click", runSCA);

    $("btnDownloadSCA")?.addEventListener("click", () => {
        if (!APP.scaCSVPath) {
            showToast("warning", "No CSV", "Run SCA analysis first.");
            return;
        }
        window.open(`/api/download/${APP.scaCSVPath}`, "_blank");
    });
}

function clearSCALayers() {
    if (APP.transectsLayer) {
        APP.map.removeLayer(APP.transectsLayer);
        APP.transectsLayer = null;
    }

    if (APP.statsLayer) {
        APP.map.removeLayer(APP.statsLayer);
        APP.statsLayer = null;
    }
}

function getTrendStyle(feature) {
    const trend = feature?.properties?.NSM_trend || "Stabil";
    return TREND_STYLES[trend] || TREND_STYLES.Stabil;
}

function bindSCAPopup(feature, layer) {
    const p = feature.properties || {};
    const html = `
        <strong>Transect ${escapeHtml(p.transect_id || "-")}</strong><br>
        NSM: ${escapeHtml(formatNum(p.NSM, 3))} m<br>
        EPR: ${escapeHtml(formatNum(p.EPR, 3))} m/year<br>
        SCE: ${escapeHtml(formatNum(p.SCE, 3))} m<br>
        LRR: ${escapeHtml(formatNum(p.LRR, 3))} m/year<br>
        Trend: ${escapeHtml(p.NSM_trend || "-")}
    `;
    layer.bindPopup(html);
}

function resetSCASummaryValues() {
    [
        "statTotalTransects",
        "statMeanNSM",
        "statMeanEPR",
        "statMeanSCE",
        "statMeanLRR",
        "statAccretionCount",
        "statErosionCount",
        "statStableCount",
        "statAutoTransectLength",
    ].forEach((id) => {
        const el = $(id);
        if (el) el.textContent = "-";
    });
}

function updateSCASummary(summary = {}) {
    const mapping = {
        statTotalTransects: summary.total_transects,
        statMeanNSM: summary.mean_NSM,
        statMeanEPR: summary.mean_EPR,
        statMeanSCE: summary.mean_SCE,
        statMeanLRR: summary.mean_LRR,
        statAccretionCount: summary.accretion_count,
        statErosionCount: summary.erosion_count,
        statStableCount: summary.stable_count,
        statAutoTransectLength: Number.isFinite(Number(summary.auto_transect_length_m))
            ? `${formatNum(summary.auto_transect_length_m, 0)} m`
            : "-",
    };

    Object.entries(mapping).forEach(([id, value]) => {
        const el = $(id);
        if (el) el.textContent = value ?? "-";
    });

    if ($("scaResults")) {
        $("scaResults").style.display = "block";
    }
}

function clearSCAResultsUIOnly() {
    clearSCALayers();
    APP.scaCSVPath = null;
    APP.scaOutputDir = null;
    resetSCASummaryValues();

    if ($("scaResults")) {
        $("scaResults").style.display = "none";
    }

    clearProgress("scaProgress", "scaProgressFill", "scaProgressText");
    updateLayerList();
    updateLegend();
}

async function deleteSCAOutputBackendOnly() {
    if (!APP.scaOutputDir) return;

    try {
        await fetch("/api/sca/delete", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                output_dir: APP.scaOutputDir,
            }),
        });
    } catch (_) {
        // silent cleanup attempt
    } finally {
        APP.scaOutputDir = null;
        APP.scaCSVPath = null;
    }
}

async function runSCA() {
    const years = Object.keys(APP.shorelines).sort();
    if (years.length < 2) {
        showToast("warning", "Need More Data", "Extract at least 2 shoreline years.");
        return;
    }

    const refYear = parseInt($("scaRefYear")?.value, 10) || parseInt(years[0], 10);

    const shorelinePaths = {};
    years.forEach((year) => {
        shorelinePaths[year] = APP.shorelines[year].path;
    });

    const params = {
        reference_year: refYear,
        baseline_offset: parseInt($("scaOffset")?.value, 10) || 300,
        transect_spacing: parseInt($("scaSpacing")?.value, 10) || 50,
        baseline_placement: $("scaPlacement")?.value || "land",
        transect_length: "auto",
    };

    setProgress("scaProgress", "scaProgressFill", "scaProgressText", 20, "Running SCA analysis...", true);
    showLoading("Running SCA analysis...");

    try {
        // cleanup previous backend sca output before running a new one
        await deleteSCAOutputBackendOnly();

        const res = await fetch("/api/sca/analyze", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                shorelines: shorelinePaths,
                params,
            }),
        });

        const data = await parseJSONResponse(res);

        setProgress("scaProgress", "scaProgressFill", "scaProgressText", 75, "Rendering transects and statistics...", true);

        clearSCALayers();

        if (hasFeatures(data.transects_geojson)) {
            APP.transectsLayer = L.geoJSON(data.transects_geojson, {
                style: {
                    color: "#8fa3b8",
                    weight: 1,
                    opacity: 0.7,
                    dashArray: "4,4",
                },
            }).addTo(APP.map);
        }

        if (hasFeatures(data.stats_geojson)) {
            APP.statsLayer = L.geoJSON(data.stats_geojson, {
                style: getTrendStyle,
                onEachFeature: bindSCAPopup,
            }).addTo(APP.map);
        }

        APP.scaCSVPath = data.sca_csv_path || data.stats_csv_path || null;
        APP.scaOutputDir = data.sca_output_dir || data.output_dir || null;

        updateSCASummary(data.summary || {});
        updateLayerList();
        updateLegend();

        const boundsSource = APP.statsLayer || APP.transectsLayer;
        if (boundsSource && boundsSource.getBounds().isValid()) {
            APP.map.fitBounds(boundsSource.getBounds(), { padding: [40, 40] });
        }

        setProgress("scaProgress", "scaProgressFill", "scaProgressText", 100, "SCA complete!", true);

        const autoLen = data?.summary?.auto_transect_length_m;
        if (Number.isFinite(Number(autoLen))) {
            showToast(
                "success",
                "SCA Completed",
                `Shoreline change analysis finished successfully. Auto transect length: ${formatNum(autoLen, 0)} m`
            );
        } else {
            showToast(
                "success",
                "SCA Completed",
                "Shoreline change analysis finished successfully."
            );
        }
    } catch (err) {
        showToast("error", "SCA Failed", err.message);
    } finally {
        hideLoading();
    }
}

/* ==========================================================
   SESSION CONTROLS
   ========================================================== */
function initSessionControls() {
    $("btnClearMapResults")?.addEventListener("click", async () => {
        const ok = confirmAction(
            "Clear map results from the interface? Backend files will be kept."
        );
        if (!ok) return;

        clearMapResultsUIOnly();
        showToast("success", "Map Cleared", "Map results have been cleared.");
    });

    $("btnResetSession")?.addEventListener("click", async () => {
        const ok = confirmAction(
            "Reset session and delete generated backend outputs? This action cannot be undone."
        );
        if (!ok) return;

        await resetSessionAndDeleteOutputs();
    });
}

function clearCompositeLayers() {
    Object.values(APP.compositeLayers).forEach((layer) => {
        if (layer && APP.map.hasLayer(layer)) {
            APP.map.removeLayer(layer);
        }
    });
    APP.compositeLayers = {};
}

function clearShorelineLayers() {
    Object.values(APP.shorelines).forEach((item) => {
        if (item?.layer && APP.map.hasLayer(item.layer)) {
            APP.map.removeLayer(item.layer);
        }
    });
    APP.shorelines = {};
}

function clearROI() {
    APP.drawnItems.clearLayers();
    APP.currentROI = null;
    APP.roiVisible = true;

    if (!APP.map.hasLayer(APP.drawnItems)) {
        APP.map.addLayer(APP.drawnItems);
    }

    updateROIInfo();
}

function clearMapResultsUIOnly() {
    clearCompositeLayers();
    clearShorelineLayers();
    clearSCAResultsUIOnly();

    updateSCAYearOptions();
    renderShorelineResults();
    updateLayerList();
    updateLegend();

    clearProgress("shorelineProgress", "shorelineProgressFill", "shorelineProgressText");
}

function resetFormsUI() {
    // dataset cards back to single default
    const datasetEntries = $("datasetEntries");
    if (datasetEntries) {
        datasetEntries.innerHTML = `
            <div class="dataset-card" data-index="0">
                <div class="dataset-header">
                    <span class="dataset-badge">Dataset 1</span>
                </div>

                <div class="form-row">
                    <div class="form-group">
                        <label>Year Label</label>
                        <input type="text" class="ds-year" value="2018" />
                    </div>
                    <div class="form-group">
                        <label>Cloud Max (%)</label>
                        <input type="number" class="ds-cloud" value="15" min="0" max="100" />
                    </div>
                </div>

                <div class="form-row">
                    <div class="form-group">
                        <label>Start Date</label>
                        <input type="date" class="ds-start" value="2018-01-01" />
                    </div>
                    <div class="form-group">
                        <label>End Date</label>
                        <input type="date" class="ds-end" value="2018-12-31" />
                    </div>
                </div>
            </div>
        `;
    }
    datasetCount = 1;

    // upload cards back to single blank/default
    const uploadEntries = $("uploadEntries");
    if (uploadEntries) {
        uploadEntries.innerHTML = `
            <div class="upload-card">
                <div class="form-group">
                    <label>Year Label</label>
                    <input type="text" class="ul-year" value="2018" />
                </div>
                <div class="file-input-wrapper">
                    <input type="file" class="ul-file" accept=".tif,.tiff" />
                </div>
            </div>
        `;
    }

    // sca inputs
    if ($("scaOffset")) $("scaOffset").value = 300;
    if ($("scaSpacing")) $("scaSpacing").value = 50;
    if ($("scaPlacement")) $("scaPlacement").value = "land";
    updateSCAYearOptions();

    clearProgress("shorelineProgress", "shorelineProgressFill", "shorelineProgressText");
    clearProgress("scaProgress", "scaProgressFill", "scaProgressText");

    if ($("shorelineResults")) $("shorelineResults").style.display = "none";
    if ($("scaResults")) $("scaResults").style.display = "none";
}

async function resetSessionAndDeleteOutputs() {
    showLoading("Resetting session and deleting backend outputs...");

    try {
        const shorelinePayload = Object.keys(APP.shorelines).map((year) => {
            const item = APP.shorelines[year];
            return {
                year_label: year,
                output_dir: item.outputDir || null,
                source_tif: item.sourceTif || null,
                geojson_path: item.path || null,
            };
        });

        const res = await fetch("/api/session/reset", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                shorelines: shorelinePayload,
                sca_output_dir: APP.scaOutputDir,
            }),
        });

        await parseJSONResponse(res);

        clearCompositeLayers();
        clearShorelineLayers();
        clearSCAResultsUIOnly();
        clearROI();
        resetFormsUI();

        updateLayerList();
        updateLegend();
        renderShorelineResults();

        showToast("success", "Session Reset", "Session cleared and backend outputs deleted.");
    } catch (err) {
        showToast("error", "Reset Failed", err.message);
    } finally {
        hideLoading();
    }
}

/* ==========================================================
   LAYER LIST
   ========================================================== */
function createLayerItem(name, color, visible, onToggle) {
    const row = document.createElement("div");
    row.className = "layer-item";

    const colorBox = document.createElement("div");
    colorBox.className = "layer-color";
    colorBox.style.background = color;

    const nameEl = document.createElement("span");
    nameEl.className = "layer-name";
    nameEl.textContent = name;

    const toggle = document.createElement("div");
    toggle.className = `layer-toggle ${visible ? "active" : ""}`;
    toggle.addEventListener("click", onToggle);

    row.appendChild(colorBox);
    row.appendChild(nameEl);
    row.appendChild(toggle);

    return row;
}

function updateLayerList() {
    const container = $("layerList");
    if (!container) return;

    container.innerHTML = "";

    if (APP.currentROI) {
        container.appendChild(
            createLayerItem(
                "ROI Boundary",
                "#00b4d8",
                APP.roiVisible,
                () => {
                    toggleROIVisibility();
                }
            )
        );
    }

    const compositeYears = Object.keys(APP.compositeLayers).sort();
    compositeYears.forEach((year) => {
        const layer = APP.compositeLayers[year];
        container.appendChild(
            createLayerItem(
                `Composite ${year}`,
                "#00b4d8",
                APP.map.hasLayer(layer),
                () => {
                    if (APP.map.hasLayer(layer)) {
                        APP.map.removeLayer(layer);
                    } else {
                        layer.addTo(APP.map);
                    }
                    updateLayerList();
                }
            )
        );
    });

    const shorelineYears = Object.keys(APP.shorelines).sort();
    shorelineYears.forEach((year) => {
        const layer = APP.shorelines[year].layer;
        container.appendChild(
            createLayerItem(
                `Shoreline ${year}`,
                APP.shorelines[year].color,
                APP.map.hasLayer(layer),
                () => {
                    if (APP.map.hasLayer(layer)) {
                        APP.map.removeLayer(layer);
                    } else {
                        layer.addTo(APP.map);
                    }
                    updateLayerList();
                }
            )
        );
    });

    if (APP.transectsLayer) {
        container.appendChild(
            createLayerItem(
                "Transects",
                "#8fa3b8",
                APP.map.hasLayer(APP.transectsLayer),
                () => {
                    if (APP.map.hasLayer(APP.transectsLayer)) {
                        APP.map.removeLayer(APP.transectsLayer);
                    } else {
                        APP.transectsLayer.addTo(APP.map);
                    }
                    updateLayerList();
                }
            )
        );
    }

    if (APP.statsLayer) {
        container.appendChild(
            createLayerItem(
                "SCA Statistics",
                "#ffd166",
                APP.map.hasLayer(APP.statsLayer),
                () => {
                    if (APP.map.hasLayer(APP.statsLayer)) {
                        APP.map.removeLayer(APP.statsLayer);
                    } else {
                        APP.statsLayer.addTo(APP.map);
                    }
                    updateLayerList();
                }
            )
        );
    }

    if (!container.children.length) {
        container.innerHTML = `<div class="layer-item"><span class="layer-name">No layers available yet</span></div>`;
    }
}

/* ==========================================================
   LEGEND
   ========================================================== */
function updateLegend() {
    const legend = $("mapLegend");
    const items = $("legendItems");
    if (!legend || !items) return;

    items.innerHTML = "";

    const shorelineYears = Object.keys(APP.shorelines).sort();
    shorelineYears.forEach((year) => {
        const row = document.createElement("div");
        row.className = "legend-item";
        row.innerHTML = `
            <span class="legend-color" style="background:${escapeHtml(APP.shorelines[year].color)};"></span>
            <span>Shoreline ${escapeHtml(year)}</span>
        `;
        items.appendChild(row);
    });

    if (APP.statsLayer) {
        [
            ["#06d6a0", "SCA Akresi"],
            ["#ef476f", "SCA Abrasi"],
            ["#ffd166", "SCA Stabil"],
        ].forEach(([color, label]) => {
            const row = document.createElement("div");
            row.className = "legend-item";
            row.innerHTML = `
                <span class="legend-color" style="background:${color};"></span>
                <span>${label}</span>
            `;
            items.appendChild(row);
        });
    }

    legend.classList.toggle("visible", items.children.length > 0);
}