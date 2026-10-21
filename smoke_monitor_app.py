"""
Sentinel-2 & Sentinel-5P Smoke Monitoring App
Persian Gulf / Strait of Hormuz smoke and air-pollution monitoring.

Sentinel-2: L2A bands + SCL -> smoke candidate mask
Sentinel-5P: CH4, CO, NO2, SO2, O3, HCHO, Aerosol Index -> maps + time series
"""
from __future__ import annotations

import io
import json
import math
import re
import hashlib
import time
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import folium
import numpy as np
import pandas as pd
import requests
import rasterio
import rasterio.transform
from rasterio.warp import transform as rio_transform
import streamlit as st
from folium.plugins import Draw, MousePosition
from scipy.ndimage import binary_dilation, gaussian_filter, label
from shapely.geometry import box, mapping, shape
from shapely.ops import unary_union
from skimage.morphology import disk
from streamlit_folium import st_folium

STAC_URL = "https://stac.dataspace.copernicus.eu/v1/"
TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
PROCESS_URL = "https://sh.dataspace.copernicus.eu/api/v1/process"

BANDS = ["B02", "B03", "B04", "B08", "B11", "B12", "SCL"]
RESOLUTION = 20
CACHE_DIR = Path.home() / ".smoke_monitor_cache"
RESULT_DIR = Path.home() / ".smoke_monitor_results"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_AOI = box(51.0, 25.0, 58.0, 29.0)

SMOKE_PARAMS = {
    "smoke_index_threshold": 0.10,
    "blue_brightness_min": 0.10,
    "swir_ratio_max": 1.60,
    "min_component_pixels": 10,
    "final_dilation": 2,
    "gaussian_sigma": 2.0,
    "max_smoke_fraction": 0.85,
}

S5P_POLLUTANTS = {
    "CH4": {"band": "CH4", "unit": "ppb", "label": "Methane (CH4)"},
    "CO": {"band": "CO", "unit": "mol/m²", "label": "Carbon monoxide (CO)"},
    "NO2": {"band": "NO2", "unit": "mol/m²", "label": "Nitrogen dioxide (NO2)"},
    "SO2": {"band": "SO2", "unit": "mol/m²", "label": "Sulfur dioxide (SO2)"},
    "O3": {"band": "O3", "unit": "mol/m²", "label": "Ozone (O3)"},
    "HCHO": {"band": "HCHO", "unit": "mol/m²", "label": "Formaldehyde (HCHO)"},
    "AER_AI": {"band": "AER_AI_340_380", "unit": "index", "label": "UV Aerosol Index"},
}


# ══════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════

def as_dict(item):
    if isinstance(item, dict):
        return item
    if hasattr(item, "to_dict"):
        return item.to_dict()
    return dict(item)


def get_properties(item):
    return as_dict(item).get("properties", {}) or {}


def get_datetime(item) -> Optional[datetime]:
    data = as_dict(item)
    props = get_properties(data)
    value = props.get("datetime") or props.get("start_datetime")
    if value:
        try:
            return datetime.fromisoformat(str(value).replace("Z", "+00:00")).replace(tzinfo=None)
        except ValueError:
            pass
    match = re.search(r"_(\d{8}T\d{6})_", data.get("id", "").upper())
    return datetime.strptime(match.group(1), "%Y%m%dT%H%M%S") if match else None


def get_tile(item):
    data = as_dict(item)
    props = get_properties(data)
    for key in ("mgrs:tile", "s2:mgrs_tile", "tile"):
        if props.get(key):
            return str(props[key]).upper()
    match = re.search(r"_(T\d{2}[A-Z]{3})_", data.get("id", "").upper())
    return match.group(1) if match else None


def get_cloud(item):
    props = get_properties(item)
    for key in ("eo:cloud_cover", "cloudCover", "cloud_cover"):
        try:
            return float(props[key])
        except Exception:
            continue
    return 100.0


def normalize_geometry(obj):
    if obj is None:
        return None
    if hasattr(obj, "__geo_interface__"):
        obj = obj.__geo_interface__
    if not isinstance(obj, dict):
        return None
    if obj.get("type") == "Feature":
        return normalize_geometry(obj.get("geometry"))
    if obj.get("type") == "FeatureCollection":
        geometries = []
        for feature in obj.get("features", []):
            geometry = normalize_geometry(feature.get("geometry"))
            if geometry:
                geometries.append(shape(geometry))
        return mapping(unary_union(geometries)) if geometries else None
    try:
        geometry = shape(obj)
        return mapping(geometry) if not geometry.is_empty else None
    except Exception:
        return None


def ensure_aoi(obj):
    return normalize_geometry(obj) or mapping(DEFAULT_AOI)


def geocode_location(query: str):
    try:
        response = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": query, "format": "json", "limit": 1, "addressdetails": 0},
            headers={"User-Agent": "SentinelSmokeMonitoringApp/1.0"},
            timeout=15,
        )
        response.raise_for_status()
        results = response.json()
        if results:
            first = results[0]
            return float(first["lat"]), float(first["lon"]), first.get("display_name", query)
    except Exception:
        pass
    return None


def search_scenes(aoi, start, end, max_cloud):
    payload = {
        "collections": ["sentinel-2-l2a"],
        "datetime": f"{start.isoformat()}Z/{end.isoformat()}Z",
        "intersects": ensure_aoi(aoi),
        "query": {"eo:cloud_cover": {"lt": float(max_cloud)}},
        "limit": 100,
    }
    response = requests.post(f"{STAC_URL}search", json=payload, timeout=120)
    response.raise_for_status()
    features = response.json().get("features", [])
    return features if isinstance(features, list) else list(features)


def authenticate_cdse(username: str, password: str, totp: str = ""):
    username, password, totp = username.strip(), password.strip(), totp.strip()
    if not username or not password:
        raise RuntimeError("Please enter your Copernicus email and password.")
    form = {
        "grant_type": "password",
        "client_id": "cdse-public",
        "username": username,
        "password": password,
    }
    if totp:
        form["totp"] = totp
    response = requests.post(TOKEN_URL, data=form, timeout=90)
    if response.status_code >= 400:
        try:
            detail = response.json().get("error_description") or response.json().get("error")
        except Exception:
            detail = None
        raise RuntimeError(detail or f"Copernicus login failed (HTTP {response.status_code}).")
    data = response.json()
    access_token = data.get("access_token")
    if not access_token:
        raise RuntimeError("Copernicus did not return an access token.")
    return {
        "access_token": access_token,
        "refresh_token": data.get("refresh_token", ""),
        "expires_at": time.time() + int(data.get("expires_in", 600)),
        "username": username,
    }


def refresh_cdse_session(auth):
    refresh_token = auth.get("refresh_token", "")
    if not refresh_token:
        return None
    response = requests.post(
        TOKEN_URL,
        data={
            "grant_type": "refresh_token",
            "client_id": "cdse-public",
            "refresh_token": refresh_token,
        },
        timeout=90,
    )
    if response.status_code >= 400:
        return None
    data = response.json()
    token = data.get("access_token")
    if not token:
        return None
    auth["access_token"] = token
    auth["refresh_token"] = data.get("refresh_token", refresh_token)
    auth["expires_at"] = time.time() + int(data.get("expires_in", 600))
    return auth


def get_access_token():
    auth = st.session_state.get("cdse_auth")
    if not auth:
        raise RuntimeError("Please log in to Copernicus first.")
    if time.time() < float(auth.get("expires_at", 0)) - 60:
        return auth["access_token"]
    refreshed = refresh_cdse_session(auth)
    if refreshed:
        st.session_state.cdse_auth = refreshed
        return refreshed["access_token"]
    st.session_state.pop("cdse_auth", None)
    raise RuntimeError("Your Copernicus session expired. Please log in again.")


# ══════════════════════════════════════════════════════════════════════════
#  SENTINEL-2 DOWNLOAD AND SMOKE MASK
# ══════════════════════════════════════════════════════════════════════════

def s2_evalscript():
    return """//VERSION=3
function setup() {
  return {
    input: [{
      bands: ["B02","B03","B04","B08","B11","B12","SCL"],
      units: ["REFLECTANCE", "REFLECTANCE", "REFLECTANCE",
              "REFLECTANCE", "REFLECTANCE", "REFLECTANCE", "DN"]
    }],
    output: {bands: 7, sampleType: "FLOAT32"}
  };
}
function evaluatePixel(sample) {
  return [sample.B02, sample.B03, sample.B04, sample.B08,
          sample.B11, sample.B12, sample.SCL];
}
"""


def request_process_tiff(payload, access_token, output_path, expected_bands):
    """Download once, validate the TIFF, and only then publish the cache file."""
    from rasterio.io import MemoryFile

    try:
        response = requests.post(
            PROCESS_URL,
            headers={"Authorization": f"Bearer {access_token}",
                     "Content-Type": "application/json"},
            json=payload,
            timeout=(30, 180),
        )
    except requests.exceptions.Timeout as exc:
        raise RuntimeError(
            "Download timed out (connection limit: 30 s; read inactivity limit: 180 s). "
            "Check your connection or try a smaller AOI, then run again."
        ) from exc
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(
            "Could not reach the CDSE Process API. Check your internet connection and retry."
        ) from exc

    if response.status_code >= 400:
        try:
            body = response.json()
            error = body.get("error", body)
            detail = error.get("message", str(error)) if isinstance(error, dict) else str(error)
        except (ValueError, AttributeError):
            detail = response.text[:1500]
        if response.status_code == 401:
            detail = "Session rejected or expired. Log out and log in to Copernicus again. " + detail
        elif response.status_code == 429:
            detail = "CDSE rate limit reached. Wait before trying again. " + detail
        raise RuntimeError(f"CDSE Process API failed ({response.status_code}): {detail}")

    try:
        with MemoryFile(response.content) as memory:
            with memory.open() as dataset:
                if dataset.count != expected_bands:
                    raise ValueError(f"Expected {expected_bands} bands, received {dataset.count}.")
                dataset.read()
    except Exception as exc:
        raise RuntimeError("CDSE response is not a readable TIFF with the expected bands.") from exc

    temporary_path = output_path.with_suffix(".tif.part")
    try:
        temporary_path.write_bytes(response.content)
        temporary_path.replace(output_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return output_path


def cached_tiff_is_valid(path, expected_bands):
    if not path.exists():
        return False
    try:
        with rasterio.open(path) as source:
            if source.count != expected_bands:
                return False
            source.read()
        return True
    except Exception:
        return False


def download_s2_scene(item, aoi, access_token):
    item = as_dict(item)
    aoi = ensure_aoi(aoi)
    cache_id = hashlib.sha256(
        json.dumps([item.get("id"), aoi, RESOLUTION, "smoke_v2_scl_dn"], sort_keys=True).encode()
    ).hexdigest()[:24]
    folder = CACHE_DIR / cache_id
    output_path = folder / "bands.tif"
    if cached_tiff_is_valid(output_path, len(BANDS)):
        return output_path

    folder.mkdir(parents=True, exist_ok=True)
    minx, miny, maxx, maxy = shape(aoi).bounds
    latitude = math.radians((miny + maxy) / 2.0)
    width = max(1, min(2500, int(abs(maxx - minx) * 111320 * math.cos(latitude) / RESOLUTION)))
    height = max(1, min(2500, int(abs(maxy - miny) * 111320 / RESOLUTION)))

    acquisition = get_datetime(item)
    if acquisition is None:
        raise RuntimeError("Could not read acquisition date.")

    payload = {
        "input": {
            "bounds": {
                "bbox": [minx, miny, maxx, maxy],
                "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"},
            },
            "data": [{
                "type": "sentinel-2-l2a",
                "dataFilter": {
                    "timeRange": {
                        "from": acquisition.strftime("%Y-%m-%dT00:00:00Z"),
                        "to": (acquisition + timedelta(days=1)).strftime("%Y-%m-%dT00:00:00Z"),
                    },
                    "mosaickingOrder": "leastCC",
                },
            }],
        },
        "output": {
            "width": width,
            "height": height,
            "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}],
        },
        "evalscript": s2_evalscript(),
    }

    return request_process_tiff(payload, access_token, output_path, len(BANDS))



def read_s2_stack(path):
    with rasterio.open(path) as source:
        array = source.read().astype(np.float32)
        profile = source.profile.copy()
    return {band: array[index] for index, band in enumerate(BANDS)}, profile


def normalized_difference(first, second):
    output = np.full(first.shape, np.nan, dtype=np.float32)
    denominator = first + second
    valid = np.isfinite(first) & np.isfinite(second) & (np.abs(denominator) > 1e-12)
    output[valid] = (first[valid] - second[valid]) / denominator[valid]
    return output


def calculate_smoke_mask(bands, params):
    b02, b03, b04 = bands["B02"], bands["B03"], bands["B04"]
    b08, b11, b12 = bands["B08"], bands["B11"], bands["B12"]
    scl = bands["SCL"]

    finite = np.logical_and.reduce([
        np.isfinite(x) for x in [b02, b03, b04, b08, b11, b12]
    ])

    scl_int = np.round(np.nan_to_num(scl, nan=0.0)).astype(np.int32)
    cloud_shadow = np.isin(scl_int, [0, 1, 3, 8, 9, 10])
    snow = scl_int == 11
    water_scl = scl_int == 6

    ndwi = normalized_difference(b03, b08)
    mndwi = normalized_difference(b03, b11)
    water = water_scl | (np.nan_to_num(mndwi, nan=-1) > 0.10)

    smoke_index = normalized_difference(b03, b11)
    swir_ratio = np.full(b11.shape, np.nan, dtype=np.float32)
    valid_ratio = np.isfinite(b11) & (b11 > 1e-6)
    swir_ratio[valid_ratio] = b12[valid_ratio] / b11[valid_ratio]

    brightness = (b02 + b03 + b04) / 3.0

    candidate = (
        finite
        & ~cloud_shadow
        & ~snow
        & ~water
        & np.isfinite(smoke_index)
        & (smoke_index > params["smoke_index_threshold"])
        & (brightness > params["blue_brightness_min"])
        & (np.nan_to_num(swir_ratio, nan=99) < params["swir_ratio_max"])
    )

    if params["gaussian_sigma"] > 0:
        smoothed = np.nan_to_num(smoke_index, nan=0.0)
        smoothed = gaussian_filter(smoothed, sigma=params["gaussian_sigma"])
        candidate &= smoothed > params["smoke_index_threshold"]

    structure = np.ones((3, 3), dtype=np.uint8)
    labeled, n_labels = label(candidate, structure=structure)
    smoke = np.zeros_like(candidate, dtype=bool)

    if n_labels > 0:
        sizes = np.bincount(labeled.ravel(), minlength=n_labels + 1)
        sizes[0] = 0
        kept = np.where(sizes >= int(params["min_component_pixels"]), sizes, 0)
        if kept.max() > 0:
            smoke = np.isin(labeled, np.where(kept > 0)[0])

    if smoke.any() and int(params["final_dilation"]) > 0:
        smoke = binary_dilation(smoke, structure=disk(int(params["final_dilation"])))

    smoke &= finite & ~cloud_shadow & ~snow & ~water

    valid = finite & ~cloud_shadow & ~snow & ~water
    smoke_fraction = float(smoke.sum() / max(1, valid.sum()))
    if smoke_fraction > params["max_smoke_fraction"]:
        smoke[:] = False
        smoke_fraction = 0.0

    diagnostics = {
        "smoke_index_max": float(np.nanmax(smoke_index)) if np.isfinite(smoke_index).any() else 0.0,
        "smoke_index_mean": float(np.nanmean(smoke_index)) if np.isfinite(smoke_index).any() else 0.0,
        "brightness_mean": float(np.nanmean(brightness)) if np.isfinite(brightness).any() else 0.0,
        "smoke_fraction": smoke_fraction,
    }

    return {
        "smoke": smoke,
        "valid": valid,
        "cloud_shadow": cloud_shadow,
        "water": water,
        "snow": snow,
        "smoke_index": smoke_index,
        "brightness": brightness,
        "diagnostics": diagnostics,
    }


def save_raster(path, array, profile, mask=False):
    output_profile = profile.copy()
    output_profile.update(
        count=1,
        dtype="uint8" if mask else "float32",
        nodata=255 if mask else -9999,
        compress="deflate",
        tiled=False,
        BIGTIFF="IF_SAFER",
    )
    output = (
        np.where(array, 1, 0).astype(np.uint8)
        if mask
        else np.where(np.isfinite(array), array, -9999).astype(np.float32)
    )
    with rasterio.open(path, "w", **output_profile) as destination:
        destination.write(output, 1)


# ══════════════════════════════════════════════════════════════════════════
#  SENTINEL-5P
# ══════════════════════════════════════════════════════════════════════════

def s5p_evalscript(band):
    return f"""//VERSION=3
function setup() {{
  return {{
    input: [{{bands: ["{band}", "dataMask"]}}],
    output: {{bands: 2, sampleType: "FLOAT32"}}
  }};
}}
function evaluatePixel(sample) {{
  return [sample.{band}, sample.dataMask];
}}
"""


def download_s5p_layer(aoi, date_from, date_to, pollutant, access_token):
    config = S5P_POLLUTANTS[pollutant]
    band = config["band"]
    aoi = ensure_aoi(aoi)
    cache_id = hashlib.sha256(
        json.dumps(["s5p", band, aoi, str(date_from), str(date_to)], sort_keys=True).encode()
    ).hexdigest()[:24]
    folder = CACHE_DIR / f"s5p_{pollutant}_{cache_id}"
    output_path = folder / "layer.tif"
    if cached_tiff_is_valid(output_path, 2):
        return output_path

    folder.mkdir(parents=True, exist_ok=True)
    minx, miny, maxx, maxy = shape(aoi).bounds
    payload = {
        "input": {
            "bounds": {
                "bbox": [minx, miny, maxx, maxy],
                "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"},
            },
            "data": [{
                "type": "sentinel-5p-l2",
                "dataFilter": {
                    "timeRange": {
                        "from": date_from.strftime("%Y-%m-%dT00:00:00Z"),
                        "to": date_to.strftime("%Y-%m-%dT23:59:59Z"),
                    }
                },
            }],
        },
        "output": {
            "width": 256,
            "height": 256,
            "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}],
        },
        "evalscript": s5p_evalscript(band),
    }

    return request_process_tiff(payload, access_token, output_path, 2)



def read_s5p(path):
    with rasterio.open(path) as source:
        values = source.read(1).astype(np.float32)
        mask = source.read(2).astype(np.float32)
    values[mask < 0.5] = np.nan
    values[~np.isfinite(values)] = np.nan
    values[values <= 0] = np.nan
    return values


# ══════════════════════════════════════════════════════════════════════════
#  VISUALIZATION
# ══════════════════════════════════════════════════════════════════════════

def create_map(aoi, search_center=None, search_zoom=6):
    geometry = shape(ensure_aoi(aoi))
    center = (
        [float(search_center[0]), float(search_center[1])]
        if search_center
        else [geometry.centroid.y, geometry.centroid.x]
    )
    zoom = int(search_zoom) if search_center else 6

    fmap = folium.Map(center, zoom_start=zoom, tiles="OpenStreetMap")
    folium.GeoJson(
        mapping(geometry),
        style_function=lambda _: {"color": "blue", "fill": False},
    ).add_to(fmap)

    if search_center:
        folium.Marker(
            [float(search_center[0]), float(search_center[1])],
            popup=folium.Popup(
                f"<b>Searched location</b><br>"
                f"Lat: {float(search_center[0]):.5f}<br>"
                f"Lon: {float(search_center[1]):.5f}",
                max_width=260,
            ),
            tooltip="Searched location",
        ).add_to(fmap)

    Draw(
        export=True,
        draw_options={
            "polyline": False,
            "circle": False,
            "marker": False,
            "circlemarker": False,
        },
    ).add_to(fmap)

    try:
        MousePosition(
            position="bottomright",
            separator=" | ",
            prefix="Lat/Lon:",
            num_digits=5,
        ).add_to(fmap)
    except Exception:
        pass

    return fmap


def smoke_png(array):
    from PIL import Image
    data = np.asarray(array)
    rgb = np.zeros((*data.shape, 3), dtype=np.uint8)
    rgb[data.astype(bool)] = [120, 120, 120]
    buffer = io.BytesIO()
    Image.fromarray(rgb).save(buffer, format="PNG")
    return buffer.getvalue()


def pollutant_png(array):
    from PIL import Image
    import matplotlib.pyplot as plt

    data = np.asarray(array).astype(np.float32)
    finite = np.isfinite(data)
    rgb = np.full((*data.shape, 3), 255, dtype=np.uint8)

    if finite.any() and finite.sum() > 1:
        values = data[finite]
        low, high = np.percentile(values, [2, 98])
        if high <= low:
            low, high = float(values.min()), float(values.max())
        if high > low:
            normalized = np.clip((np.nan_to_num(data, nan=low) - low) / (high - low), 0, 1)
            rgb = (plt.get_cmap("inferno")(normalized)[:, :, :3] * 255).astype(np.uint8)
            rgb[~finite] = 255

    buffer = io.BytesIO()
    Image.fromarray(rgb).save(buffer, format="PNG")
    return buffer.getvalue()


def legend_html(kind):
    if kind == "smoke":
        rows = [("#787878", "Smoke candidate"), ("#ffffff", "Background / masked")]
    elif kind == "valid":
        rows = [("#dc1e1e", "Valid pixels"), ("#000000", "Invalid / masked pixels")]
    else:
        rows = [
            ("#fcffa4", "Higher value"),
            ("#bc3754", "Medium value"),
            ("#000004", "Lower value"),
            ("#ffffff", "No data"),
        ]
    items = "".join(
        f'<div class="legend-row"><span class="legend-swatch" style="background:{c};"></span><span>{t}</span></div>'
        for c, t in rows
    )
    return f'<div class="result-legend"><div class="legend-heading">Legend</div>{items}</div>'


def georeferenced_png_package(array, profile, mask=False):
    png_data = smoke_png(array) if mask else pollutant_png(array)
    transform = profile["transform"]
    xres, yres = transform.a, transform.e
    x_center = transform.c + xres / 2.0
    y_center = transform.f + yres / 2.0
    pgw = f"{xres:.12f}\n0.0\n0.0\n{yres:.12f}\n{x_center:.12f}\n{y_center:.12f}\n"
    crs_text = profile.get("crs")
    prj = crs_text.to_wkt() if crs_text else ""

    package = io.BytesIO()
    with zipfile.ZipFile(package, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("output.png", png_data)
        archive.writestr("output.pgw", pgw.encode("utf-8"))
        if prj:
            archive.writestr("output.prj", prj.encode("utf-8"))
    return package.getvalue()


# ══════════════════════════════════════════════════════════════════════════
#  UI
# ══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Sentinel Smoke Monitor",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
:root {
    --red: #e63946;
    --honeydew: #f1faee;
    --frost: #a8dadc;
    --blue: #457b9d;
    --navy: #1d3557;
    --black: #111111;
    --white: #ffffff;
    --border: #d8e6e8;
    --muted: #4f5d63;
    --dark-field: #292a33;
}
.stApp { background: #f1faee; color: #111111 !important; }
[data-testid="stHeader"] { background: #f1faee !important; height: 3.25rem !important; }
[data-testid="stSidebar"] { display: none; }
.block-container { max-width: 1700px; padding-top: 3.9rem !important; padding-bottom: 0.8rem; padding-left: 1.2rem; padding-right: 1.2rem; }
.app-header { position: relative; z-index: 10; display: flex; align-items: center; justify-content: space-between; background: #ffffff; border: 1px solid var(--border); border-radius: 16px; padding: 0.75rem 1rem; margin-top: 0.15rem; margin-bottom: 0.9rem; box-shadow: 0 2px 10px rgba(29,53,87,0.05); }
.app-title { color: #111111 !important; font-size: 1.45rem; font-weight: 850; line-height: 1.1; }
.app-subtitle { color: #111111 !important; font-size: 0.78rem; margin-top: 0.15rem; }
.status-pill { background: #f1faee; color: #111111 !important; border: 1px solid #a8dadc; border-radius: 999px; padding: 0.35rem 0.7rem; font-size: 0.72rem; font-weight: 750; white-space: nowrap; }
.app-card { background: #ffffff; border: 1px solid var(--border); border-radius: 15px; padding: 0.75rem; box-shadow: 0 2px 10px rgba(29,53,87,0.04); height: 100%; color: #111111 !important; }
.card-title { color: #111111 !important; font-size: 1rem; font-weight: 800; margin-bottom: 0.1rem; }
.card-caption { color: #111111 !important; font-size: 0.73rem; margin-bottom: 0.45rem; }
.section-label { display: inline-block; background: #a8dadc; color: #111111 !important; border-radius: 999px; padding: 0.2rem 0.55rem; font-size: 0.65rem; font-weight: 800; letter-spacing: 0.03em; margin-bottom: 0.35rem; }
.stApp p, .stApp label, .stApp small, .stApp strong, .stApp em, .stApp li, .stApp td, .stApp th,
.stApp [data-testid="stMarkdownContainer"], .stApp [data-testid="stMarkdownContainer"] p,
.stApp [data-testid="stMarkdownContainer"] span, .stApp [data-testid="stMarkdownContainer"] li { color: #111111 !important; }
div[data-testid="stDateInput"] input, .stDateInput input,
div[data-testid="stNumberInput"] input, .stNumberInput input {
    background-color: var(--dark-field) !important; color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important; caret-color: #ffffff !important;
}
input::placeholder, textarea::placeholder { color: #bfc3cc !important; opacity: 1 !important; }
div[data-baseweb="select"] input, div[data-baseweb="select"] [role="combobox"], div[data-baseweb="select"] * { color: #111111 !important; }
div[data-baseweb="popover"] [role="listbox"], div[data-baseweb="popover"] ul[role="listbox"],
div[data-baseweb="popover"] [role="option"], div[data-baseweb="popover"] li[role="option"] { background: #111318 !important; }
div[data-baseweb="popover"] [role="option"] *, ul[role="listbox"] *, li[role="option"] * { color: #ffffff !important; -webkit-text-fill-color: #ffffff !important; }
div[data-baseweb="popover"] [role="option"]:hover, div[data-baseweb="popover"] li[role="option"]:hover { background: #2b2e38 !important; }
.stDateInput, .stSlider, .stNumberInput, .stSelectbox { margin-bottom: 0.15rem; }
.stSlider label, .stSlider [data-testid="stTickBar"] * { color: #111111 !important; }
.stSlider [data-testid="stThumbValue"], .stSlider [data-testid="stThumbValue"] * { color: #ffffff !important; }
[data-baseweb="calendar"] *, [data-baseweb="popover"] [data-baseweb="calendar"] *, [data-baseweb="calendar"] button { color: #ffffff !important; }
input:-webkit-autofill, input:-webkit-autofill:hover, input:-webkit-autofill:focus { -webkit-text-fill-color: #ffffff !important; caret-color: #ffffff !important; }
.auth-card { background: #f8fbfb; border: 1px solid #d7e4e7; border-radius: 11px; padding: 0.65rem 0.75rem; margin-top: 0.45rem; }
.auth-status { background: #e8f7ea; border: 1px solid #9ed2a4; color: #155724 !important; border-radius: 9px; padding: 0.45rem 0.6rem; font-size: 0.76rem; font-weight: 700; margin-bottom: 0.45rem; }
.auth-help { color: #111111 !important; font-size: 0.72rem; line-height: 1.45; margin: 0.2rem 0 0.45rem 0; }
.stButton > button, .stDownloadButton > button { border-radius: 9px; min-height: 2.15rem; font-weight: 750; font-size: 0.78rem; color: #111111 !important; }
.stButton > button[kind="primary"] { background: #e63946; border-color: #e63946; color: #ffffff !important; }
.stButton > button[kind="primary"] *, .stDownloadButton > button[kind="primary"] * { color: #ffffff !important; }
.stButton > button[kind="primary"]:hover { background: #c92f3b; border-color: #c92f3b; color: #ffffff !important; }
.stDownloadButton > button { background: #ffffff; color: #111111 !important; border: 1px solid #a8dadc; }
.stDownloadButton > button:hover { background: #f1faee; border-color: #457b9d; color: #111111 !important; }
div[data-testid="stDataFrame"] { border: 1px solid var(--border); }
div[data-testid="stDataFrame"] * { color: #111111 !important; }
.result-card { background: #ffffff; border: 1px solid #d7e4e7; border-radius: 12px; padding: 0.7rem; height: 100%; }
.result-tag { display: inline-block; background: #a8dadc; color: #111111 !important; border-radius: 999px; padding: 0.15rem 0.5rem; font-size: 0.62rem; font-weight: 800; margin-bottom: 0.25rem; }
.result-name { color: #111111 !important; font-size: 0.85rem; font-weight: 800; margin-bottom: 0.35rem; }
.result-note { background: #ffffff; border: 1px solid #d7e4e7; border-radius: 10px; padding: 0.7rem 0.8rem; color: #111111 !important; font-size: 0.76rem; line-height: 1.55; margin-top: 0.5rem; }
.download-label { color: #111111 !important; font-size: 0.62rem; font-weight: 700; margin: 0.2rem 0 0.12rem 0; }
.map-frame { border: 1px solid var(--border); border-radius: 10px; overflow: hidden; }
.result-legend { background: #ffffff; border: 1px solid #d7e4e7; border-radius: 10px; padding: 0.75rem 0.7rem; min-height: 96px; box-sizing: border-box; display: flex; flex-direction: column; justify-content: center; gap: 0.42rem; }
.result-legend .legend-heading { color: #111111 !important; font-size: 0.88rem; font-weight: 800; }
.result-legend .legend-row { display: flex; align-items: center; gap: 0.45rem; color: #111111 !important; font-size: 0.82rem; line-height: 1.25; }
.result-legend .legend-row span:last-child { color: #111111 !important; }
.legend-swatch { width: 18px; height: 14px; min-width: 18px; border: 1px solid #555; border-radius: 2px; display: inline-block; }
.coords-panel { margin-top: 0.55rem; background: #f8fbfb; border: 1px solid #d7e4e7; border-radius: 10px; padding: 0.5rem 0.7rem; display: flex; flex-direction: column; gap: 0.3rem; }
.coord-row { display: flex; justify-content: space-between; align-items: center; gap: 0.5rem; font-size: 0.78rem; color: #111111 !important; }
.coord-label { font-weight: 700; color: #111111 !important; flex: 1 1 auto; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.coord-value { flex: 0 0 auto; font-family: ui-monospace, "SFMono-Regular", Menlo, Consolas, monospace; background: #eef5f6; border: 1px solid #cfe0e3; border-radius: 6px; padding: 0.12rem 0.5rem; color: #111111 !important; font-size: 0.74rem; white-space: nowrap; }
footer { visibility: hidden; }
.stMarkdown { margin-bottom: 0.1rem; }
.element-container { margin-bottom: 0.15rem; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="app-header">
    <div>
        <div class="app-title">🛰️ Sentinel Smoke & Air Pollution Monitor</div>
        <div class="app-subtitle">Sentinel-2 smoke masking &nbsp;|&nbsp; Sentinel-5P multi-pollutant time series &nbsp;|&nbsp; Persian Gulf &amp; Strait of Hormuz</div>
    </div>
    <div class="status-pill">20 m S2 processing &nbsp;&nbsp; Multi-pollutant S5P</div>
</div>
""", unsafe_allow_html=True)

if "aoi" not in st.session_state:
    st.session_state.aoi = mapping(DEFAULT_AOI)

map_col, control_col = st.columns([1.65, 1.0], gap="small")

with map_col:
    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">01 · STUDY AREA</div>', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Area of Interest</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="card-caption">Search a location or draw the study area directly on the map.</div>',
        unsafe_allow_html=True,
    )

    search_col1, search_col2 = st.columns([4, 1], gap="small")
    with search_col1:
        search_query = st.text_input(
            "Location search",
            placeholder="🔎  Search a place  (e.g. Strait of Hormuz, Bandar Abbas, Bushehr, …)",
            key="location_search_input",
            label_visibility="collapsed",
        )
    with search_col2:
        search_clicked = st.button("🔍 Find", use_container_width=True, key="search_location_btn")

    if search_clicked:
        query = (search_query or "").strip()
        if not query:
            st.warning("Please type a location name first.")
        else:
            with st.spinner("Searching location…"):
                geo = geocode_location(query)
            if geo:
                st.session_state["search_center"] = [geo[0], geo[1]]
                st.session_state["search_name"] = geo[2]
                st.session_state["search_zoom"] = 9
            else:
                st.warning("Location not found. Try a different name or spelling.")

    if st.session_state.get("search_center"):
        info_col, clear_col = st.columns([4, 1], gap="small")
        with info_col:
            st.markdown(
                f'<div class="card-caption" style="margin-top:0.35rem;">'
                f'📍 {st.session_state.get("search_name", "Searched location")}</div>',
                unsafe_allow_html=True,
            )
        with clear_col:
            if st.button("✖ Clear", use_container_width=True, key="clear_search_btn"):
                st.session_state.pop("search_center", None)
                st.session_state.pop("search_name", None)
                st.rerun()

    map_center = st.session_state.get("search_center")
    map_zoom = st.session_state.get("search_zoom", 6)
    map_data = st_folium(
        create_map(st.session_state.aoi, search_center=map_center, search_zoom=map_zoom),
        height=385,
        width=1000,
        key="aoi_map",
    )

    if map_data and map_data.get("all_drawings"):
        new_aoi = normalize_geometry(
            {"type": "FeatureCollection", "features": map_data["all_drawings"]}
        )
        if new_aoi:
            st.session_state.aoi = new_aoi

    aoi_geom = shape(ensure_aoi(st.session_state.aoi))
    aoi_c = aoi_geom.centroid
    coord_lines = [
        f'<div class="coord-row"><span class="coord-label">🎯 AOI center</span>'
        f'<span class="coord-value">Lat {aoi_c.y:.5f} · Lon {aoi_c.x:.5f}</span></div>'
    ]

    if st.session_state.get("search_center"):
        sc = st.session_state["search_center"]
        sname = st.session_state.get("search_name", "Searched location")
        short_name = (sname[:70] + "…") if len(sname) > 70 else sname
        coord_lines.append(
            f'<div class="coord-row"><span class="coord-label">📍 {short_name}</span>'
            f'<span class="coord-value">Lat {sc[0]:.5f} · Lon {sc[1]:.5f}</span></div>'
        )

    if map_data and map_data.get("last_clicked"):
        lc = map_data["last_clicked"]
        coord_lines.append(
            f'<div class="coord-row"><span class="coord-label">🖱️ Last click on map</span>'
            f'<span class="coord-value">Lat {lc["lat"]:.5f} · Lon {lc["lng"]:.5f}</span></div>'
        )

    st.markdown('<div class="coords-panel">' + "".join(coord_lines) + "</div>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with control_col:
    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">02 · SEARCH</div>', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Scene Search</div>', unsafe_allow_html=True)

    default_end = datetime.now().date()
    default_start = default_end - timedelta(days=30)
    d1, d2 = st.columns(2, gap="small")
    with d1:
        start_date = st.date_input("Start date", default_start, key="start_date")
    with d2:
        end_date = st.date_input("End date", default_end, key="end_date")

    s1, s2 = st.columns(2, gap="small")
    with s1:
        max_cloud = st.slider("Cloud cover (%)", 0.0, 100.0, 40.0, key="max_cloud")
    with s2:
        s5p_window = st.slider("S5P temporal window (days)", 1, 30, 5, key="s5p_window")

    if st.button("🔍  Search Sentinel-2 scenes", type="primary", use_container_width=True):
        try:
            with st.spinner("Searching CDSE STAC..."):
                search_results = search_scenes(
                    st.session_state.aoi,
                    datetime.combine(start_date, datetime.min.time()),
                    datetime.combine(end_date, datetime.max.time()),
                    max_cloud,
                )
            st.session_state["scene_results"] = search_results
            st.session_state.pop("target", None)
            if search_results:
                st.success(f"{len(search_results)} scene(s) found")
            else:
                st.warning("No Sentinel-2 scenes were found for the selected criteria.")
        except Exception as error:
            st.session_state["scene_results"] = []
            st.session_state.pop("target", None)
            st.error(f"Scene search failed: {error}")

    scene_results = st.session_state.get("scene_results", [])

    if scene_results:
        scene_table = pd.DataFrame([
            {
                "date": get_datetime(scene),
                "tile": get_tile(scene),
                "cloud": get_cloud(scene),
            }
            for scene in scene_results
        ]).sort_values(["date", "cloud"], ascending=[True, True], na_position="last")

        st.dataframe(
            scene_table,
            use_container_width=True,
            height=112,
            hide_index=True,
            column_config={
                "date": st.column_config.DatetimeColumn("Date", format="YYYY-MM-DD"),
                "cloud": st.column_config.NumberColumn("Cloud %", format="%.1f"),
            },
        )

        scene_ids = [as_dict(scene).get("id") for scene in scene_results if as_dict(scene).get("id")]

        def format_scene(scene_id):
            scene = next(
                (candidate for candidate in scene_results if as_dict(candidate).get("id") == scene_id),
                None,
            )
            if scene is None:
                return str(scene_id)
            scene_date = get_datetime(scene)
            date_text = scene_date.strftime("%Y-%m-%d") if scene_date else "Unknown date"
            return f"{date_text}  |  {get_tile(scene) or 'Unknown tile'}  |  cloud {get_cloud(scene):.1f}%"

        if scene_ids:
            selected_scene_id = st.selectbox(
                "Target scene",
                scene_ids,
                format_func=format_scene,
                key="target_scene_select",
            )
            selected_scene = next(
                (scene for scene in scene_results if as_dict(scene).get("id") == selected_scene_id),
                None,
            )
            if selected_scene is not None:
                st.session_state["target"] = selected_scene

    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div style="height:0.25rem"></div>', unsafe_allow_html=True)

settings_col, action_col = st.columns([1.65, 1.0], gap="small")

with settings_col:
    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">03 · SMOKE DETECTION</div>', unsafe_allow_html=True)

    p1, p2, p3 = st.columns(3, gap="small")
    with p1:
        SMOKE_PARAMS["smoke_index_threshold"] = st.number_input(
            "Smoke index threshold",
            min_value=0.01, max_value=0.80,
            value=float(SMOKE_PARAMS["smoke_index_threshold"]),
            step=0.01, key="smoke_index_threshold",
        )
    with p2:
        SMOKE_PARAMS["min_component_pixels"] = st.number_input(
            "Minimum smoke pixels",
            min_value=2, max_value=1000,
            value=int(SMOKE_PARAMS["min_component_pixels"]),
            step=5, key="min_component_pixels",
        )
    with p3:
        SMOKE_PARAMS["final_dilation"] = st.number_input(
            "Final dilation radius",
            min_value=0, max_value=20,
            value=int(SMOKE_PARAMS["final_dilation"]),
            step=1, key="final_dilation",
        )

    estimated_area_m2 = int(SMOKE_PARAMS["min_component_pixels"]) * RESOLUTION * RESOLUTION
    st.markdown(
        f'<div class="card-caption">Minimum connected smoke region ≈ {estimated_area_m2:,} m² at {RESOLUTION} m resolution.</div>',
        unsafe_allow_html=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)

with action_col:
    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">04 · PROCESS</div>', unsafe_allow_html=True)

    cdse_auth = st.session_state.get("cdse_auth")
    if cdse_auth:
        st.markdown(
            f'<div class="auth-status">✓ Copernicus connected · {cdse_auth.get("username", "")}</div>',
            unsafe_allow_html=True,
        )
        logout_col, _ = st.columns([1, 2])
        with logout_col:
            if st.button("Log out", use_container_width=True, key="cdse_logout"):
                st.session_state.pop("cdse_auth", None)
                st.rerun()
    else:
        st.markdown('<div class="auth-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Copernicus login</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="auth-help">Log in once in this browser session. Your password is sent directly to the official Copernicus identity service; the app keeps only the temporary API token.</div>',
            unsafe_allow_html=True,
        )
        st.link_button("🌐 Open Copernicus website", "https://dataspace.copernicus.eu/", use_container_width=True)

        with st.form("cdse_login_form", clear_on_submit=True):
            login_user = st.text_input("Copernicus email", placeholder="your-email@example.com")
            login_password = st.text_input("Copernicus password", type="password")
            login_totp = st.text_input("2FA code (optional)", max_chars=8, placeholder="Only if your account uses 2FA")
            login_submitted = st.form_submit_button("🔐 Login & connect", type="primary", use_container_width=True)

        if login_submitted:
            try:
                with st.spinner("Connecting to Copernicus…"):
                    st.session_state.cdse_auth = authenticate_cdse(login_user, login_password, login_totp)
                st.success("Copernicus login successful.")
                st.rerun()
            except Exception as login_error:
                st.error(str(login_error))
        st.markdown('</div>', unsafe_allow_html=True)

    if scene_results and "target" in st.session_state:
        target = st.session_state.get("target")
        if target is not None:
            target_date = get_datetime(target)
            target_tile = get_tile(target) or "Unknown tile"
            target_label = target_date.strftime("%Y-%m-%d") if target_date else "Unknown date"
            st.markdown(
                f'<div class="card-title">Ready to detect smoke</div>'
                f'<div class="card-caption">Target: {target_label} · {target_tile}</div>',
                unsafe_allow_html=True,
            )
            detect_clicked = st.button(
                "🛰️  Download AOI & Detect Smoke",
                type="primary",
                use_container_width=True,
                key="detect_button",
                disabled=not bool(st.session_state.get("cdse_auth")),
            )

            if not st.session_state.get("cdse_auth"):
                st.markdown(
                    '<div class="card-caption">Please connect your Copernicus account above before downloading Sentinel-2 data.</div>',
                    unsafe_allow_html=True,
                )

            if detect_clicked:
                progress = st.progress(0, text="Preparing smoke detection…")
                progress_status = st.empty()
                try:
                    progress_status.markdown(
                        '<div class="card-caption">Step 1 of 4 · Downloading Sentinel-2 bands…</div>',
                        unsafe_allow_html=True,
                    )
                    progress.progress(15, text="Downloading Sentinel-2 bands…")
                    access_token = get_access_token()
                    target = st.session_state["target"]
                    target_date = get_datetime(target)
                    if target_date is None:
                        raise RuntimeError("The selected target scene has no valid acquisition date.")

                    s2_path = download_s2_scene(target, st.session_state.aoi, access_token)
                    bands, profile = read_s2_stack(s2_path)

                    progress_status.markdown(
                        '<div class="card-caption">Step 2 of 4 · Calculating smoke index and quality masks…</div>',
                        unsafe_allow_html=True,
                    )
                    progress.progress(50, text="Calculating smoke mask…")
                    result = calculate_smoke_mask(bands, SMOKE_PARAMS)

                    progress_status.markdown(
                        '<div class="card-caption">Step 3 of 4 · Saving georeferenced outputs…</div>',
                        unsafe_allow_html=True,
                    )
                    progress.progress(80, text="Saving outputs…")
                    output_folder = RESULT_DIR / target_date.strftime("%Y%m%d")
                    output_folder.mkdir(parents=True, exist_ok=True)
                    paths = {}
                    for key in ("smoke", "valid", "smoke_index"):
                        paths[key] = output_folder / f"{key}.tif"
                        save_raster(
                            paths[key],
                            result[key],
                            profile,
                            mask=key in ("smoke", "valid"),
                        )

                    st.session_state.smoke_result = result
                    st.session_state.smoke_paths = paths
                    st.session_state.smoke_profile = profile
                    st.session_state.smoke_pngs = {
                        "smoke": smoke_png(result["smoke"]),
                        "valid": smoke_png(result["valid"]),
                        "smoke_index": pollutant_png(result["smoke_index"]),
                    }

                    progress_status.markdown(
                        '<div class="card-caption">Step 4 of 4 · Preparing downloads…</div>',
                        unsafe_allow_html=True,
                    )
                    progress.progress(100, text="Smoke detection completed")
                    st.success("Smoke detection completed")

                    if result["smoke"].sum() == 0:
                        st.warning("⚠️ No smoke candidate was detected with the current settings.")
                except Exception as error:
                    progress.empty()
                    progress_status.empty()
                    st.error(f"Smoke detection failed: {error}")
                else:
                    progress_status.empty()
    else:
        st.markdown(
            '<div class="card-title">Select scenes first</div>'
            '<div class="card-caption">Search for Sentinel-2 scenes, select a target, then run smoke detection.</div>',
            unsafe_allow_html=True,
        )
    st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
#  RESULTS
# ══════════════════════════════════════════════════════════════════════════

if "smoke_result" in st.session_state:
    result = st.session_state.smoke_result
    png_outputs = st.session_state.smoke_pngs
    profile = st.session_state.smoke_profile

    st.markdown('<div style="height:0.25rem"></div>', unsafe_allow_html=True)
    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">05 · RESULTS</div>', unsafe_allow_html=True)

    metrics = st.columns(5, gap="small")
    metrics[0].metric("Smoke pixels", f"{int(result['smoke'].sum()):,}")
    metrics[1].metric("Valid pixels", f"{int(result['valid'].sum()):,}")
    metrics[2].metric("Smoke fraction", f"{result['diagnostics']['smoke_fraction'] * 100:.3f}%")
    metrics[3].metric("Max smoke index", f"{result['diagnostics']['smoke_index_max']:.3f}")
    metrics[4].metric("Mean brightness", f"{result['diagnostics']['brightness_mean']:.3f}")

    result_items = [
        ("smoke", "Smoke candidate mask", "Smoke mask"),
        ("smoke_index", "Smoke index", "Smoke index"),
        ("valid", "Valid pixels", "Validity mask"),
    ]
    result_cols = st.columns(3, gap="small")
    for col, (key, title, tag) in zip(result_cols, result_items):
        with col:
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.markdown(f'<div class="result-tag">{tag}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="result-name">{title}</div>', unsafe_allow_html=True)
            preview_col, legend_col = st.columns([3.6, 1.0], gap="small")
            with preview_col:
                st.image(png_outputs[key], use_container_width=True, output_format="PNG")
            with legend_col:
                st.markdown('<div style="padding-top:0.35rem;"></div>', unsafe_allow_html=True)
                st.markdown(legend_html("smoke" if key == "smoke" else "valid" if key == "valid" else "pollutant"), unsafe_allow_html=True)

            path = st.session_state.smoke_paths[key]
            format_choice = st.selectbox(
                "Download format",
                ["GeoTIFF (georeferenced)", "PNG + World File (georeferenced)"],
                key=f"format_choice_{key}",
            )
            if format_choice == "GeoTIFF (georeferenced)":
                st.download_button(
                    "⬇ Download GeoTIFF",
                    path.read_bytes(),
                    file_name=path.name,
                    mime="image/tiff",
                    key=f"download_tif_{key}",
                    use_container_width=True,
                )
            else:
                package = georeferenced_png_package(
                    result[key],
                    profile,
                    mask=key in ("smoke", "valid"),
                )
                st.download_button(
                    "⬇ Download Georeferenced PNG package",
                    package,
                    file_name=f"{key}_georeferenced_png.zip",
                    mime="application/zip",
                    key=f"download_png_{key}",
                    use_container_width=True,
                )
            st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(
        f'<div class="result-note"><b>Smoke detection:</b> smoke index = (B03 − B11) / (B03 + B11). '
        f'Cloud, shadow, snow and water pixels are excluded using Sentinel-2 L2A SCL. '
        f'Initial threshold = <b>{SMOKE_PARAMS["smoke_index_threshold"]:.2f}</b>; '
        f'minimum connected pixels = <b>{int(SMOKE_PARAMS["min_component_pixels"])}</b>; '
        f'final dilation = <b>{int(SMOKE_PARAMS["final_dilation"])}</b>.</div>',
        unsafe_allow_html=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
#  S5P MULTI-POLLUTANT
# ══════════════════════════════════════════════════════════════════════════

st.markdown('<div style="height:0.35rem"></div>', unsafe_allow_html=True)
st.markdown('<div class="app-card">', unsafe_allow_html=True)
st.markdown('<div class="section-label">05a · SENTINEL-5P POLLUTANTS</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="card-caption">Select one or more Sentinel-5P pollutants. The app downloads each selected layer around the target date, shows its map and builds a time series.</div>',
    unsafe_allow_html=True,
)

selected_pollutants = st.multiselect(
    "Pollutants",
    list(S5P_POLLUTANTS.keys()),
    default=["CO", "NO2", "SO2", "AER_AI"],
    format_func=lambda key: S5P_POLLUTANTS[key]["label"],
    key="selected_pollutants",
)

s5p_col1, s5p_col2 = st.columns([1, 3], gap="small")
with s5p_col1:
    run_s5p = st.button(
        "🛰️  Fetch S5P layers",
        type="primary",
        use_container_width=True,
        key="s5p_button",
        disabled=not bool(st.session_state.get("cdse_auth")) or not selected_pollutants,
    )
with s5p_col2:
    st.markdown(
        '<div class="card-caption">Each selected pollutant is downloaded for the S5P temporal window around the selected Sentinel-2 target date.</div>',
        unsafe_allow_html=True,
    )

if run_s5p:
    s5p_progress = None
    try:
        access_token = get_access_token()
        target_date = get_datetime(st.session_state.get("target"))
        if target_date is None:
            raise RuntimeError("Target date is missing.")

        date_from = target_date - timedelta(days=int(s5p_window))
        date_to = target_date + timedelta(days=int(s5p_window))

        s5p_data = {}
        s5p_progress = st.progress(0, text="Fetching Sentinel-5P layers…")
        for index, pollutant in enumerate(selected_pollutants, start=1):
            s5p_progress.progress(
                int(100 * index / len(selected_pollutants)),
                text=f"Fetching {S5P_POLLUTANTS[pollutant]['label']}…",
            )
            path = download_s5p_layer(
                st.session_state.aoi,
                date_from,
                date_to,
                pollutant,
                access_token,
            )
            s5p_data[pollutant] = read_s5p(path)

        st.session_state.s5p_data = s5p_data
        st.session_state.s5p_date_range = (date_from, date_to)
        s5p_progress.progress(100, text="Sentinel-5P layers ready")
        st.success(f"Sentinel-5P layers loaded · {len(s5p_data)} pollutant(s)")
    except Exception as s5p_error:
        if s5p_progress is not None:
            s5p_progress.empty()
        st.error(f"S5P fetch failed: {s5p_error}")

if "s5p_data" in st.session_state:
    s5p_data = st.session_state.s5p_data
    pollutant_cols = st.columns(min(3, len(s5p_data)) if s5p_data else 1, gap="small")

    for index, (pollutant, values) in enumerate(s5p_data.items()):
        with pollutant_cols[index % len(pollutant_cols)]:
            config = S5P_POLLUTANTS[pollutant]
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.markdown(f'<div class="result-tag">{config["label"]}</div>', unsafe_allow_html=True)
            valid_values = values[np.isfinite(values)]
            if valid_values.size:
                st.metric(
                    f"Mean {pollutant}",
                    f"{float(np.nanmean(valid_values)):.3g} {config['unit']}",
                )
                st.image(pollutant_png(values), use_container_width=True, output_format="PNG")
            else:
                st.warning("No valid pixels for this pollutant in the selected window.")
            st.markdown('</div>', unsafe_allow_html=True)

    if st.button(
        "📈  Build S5P time series",
        type="primary",
        use_container_width=True,
        key="build_s5p_ts",
        disabled=not bool(st.session_state.get("cdse_auth")) or not selected_pollutants,
    ):
        try:
            access_token = get_access_token()
            target_date = get_datetime(st.session_state.get("target"))
            if target_date is None:
                raise RuntimeError("Target date is missing.")

            date_from = target_date - timedelta(days=int(s5p_window))
            date_to = target_date + timedelta(days=int(s5p_window))
            days = pd.date_range(date_from.date(), date_to.date(), freq="D")

            rows = []
            ts_progress = st.progress(0, text="Building S5P time series…")
            total_steps = max(1, len(days) * len(selected_pollutants))
            step = 0

            for day in days:
                row = {"date": day}
                for pollutant in selected_pollutants:
                    try:
                        path = download_s5p_layer(
                            st.session_state.aoi,
                            day,
                            day,
                            pollutant,
                            access_token,
                        )
                        values = read_s5p(path)
                        valid = values[np.isfinite(values)]
                        row[pollutant] = float(np.nanmean(valid)) if valid.size else np.nan
                    except Exception:
                        row[pollutant] = np.nan
                    step += 1
                    ts_progress.progress(
                        int(100 * step / total_steps),
                        text=f"{day.date()} · {pollutant}",
                    )
                rows.append(row)

            ts_df = pd.DataFrame(rows).set_index("date")
            st.session_state.s5p_timeseries = ts_df
            ts_progress.progress(100, text="S5P time series ready")
            st.success("S5P time series built")
        except Exception as ts_error:
            st.error(f"S5P time series failed: {ts_error}")

    if "s5p_timeseries" in st.session_state:
        ts = st.session_state.s5p_timeseries
        if not ts.empty:
            st.markdown(
                '<div class="card-caption">Daily mean values over the AOI for selected Sentinel-5P pollutants</div>',
                unsafe_allow_html=True,
            )
            st.line_chart(ts, use_container_width=True, height=280)
            st.dataframe(ts, use_container_width=True, hide_index=True, height=200)
            st.download_button(
                "⬇ Download S5P time series CSV",
                ts.to_csv(),
                file_name="sentinel5p_pollutants_timeseries.csv",
                mime="text/csv",
                key="download_s5p_ts_csv",
                use_container_width=False,
            )

st.markdown('</div>', unsafe_allow_html=True)
