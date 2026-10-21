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
from rasterio.warp import transform_bounds
from rasterio.features import geometry_mask
from rasterio.io import MemoryFile
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

BANDS = ["B02", "B03", "B04", "B08", "B11", "B12", "SCL", "dataMask", "observation_day"]
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
    if item is None:
        return {}
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
    if start > end:
        raise RuntimeError("Start date must be before end date.")
    payload={"collections":["sentinel-2-l2a"],
             "datetime":f"{start.isoformat()}Z/{end.isoformat()}Z",
             "intersects":ensure_aoi(aoi),"limit":100}
    features=[]
    seen=set()
    url=f"{STAC_URL}search"
    method="POST"
    for page in range(100):
        if method=="POST":
            response=requests.post(url,json=payload,timeout=(30,120))
        else:
            response=requests.get(url,timeout=(30,120))
        response.raise_for_status()
        data=response.json()
        batch=data.get("features",[])
        added=0
        for item in batch:
            identity=item.get("id")
            if identity not in seen:
                features.append(item)
                seen.add(identity)
                added+=1
        nxt=next((link for link in data.get("links",[]) if link.get("rel")=="next"),None)
        if not nxt:
            break
        if added==0:
            raise RuntimeError("STAC pagination repeated a page; narrow the date range to avoid incomplete results.")
        from urllib.parse import urljoin
        url=urljoin(url,nxt["href"])
        method=nxt.get("method","GET").upper()
        if method=="POST":
            body=nxt.get("body",{})
            payload={**payload,**body} if nxt.get("merge",False) else (body or payload)
    else:
        raise RuntimeError("Too many STAC pages. Search a shorter date interval.")
    return features


def authenticate_cdse(username: str, password: str, totp: str = ""):
    username, totp = username.strip(), totp.strip()
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


def processing_options():
    state = st.session_state
    return {
        "resolution": int(state.get("regional_resolution", 250)),
        "window": int(state.get("s2_composite_days", 0)),
        "keep_medium": bool(state.get("keep_medium_cloud", False)),
        "qa_override": bool(state.get("s5p_qa_override", False)),
        "qa": int(state.get("s5p_min_qa", 50)),
    }


def grid_for_aoi(aoi):
    geom = shape(ensure_aoi(aoi))
    lon, lat = geom.centroid.x, geom.centroid.y
    zone = min(60, max(1, int((lon + 180) // 6) + 1))
    epsg = (32600 if lat >= 0 else 32700) + zone
    bounds = transform_bounds("EPSG:4326", f"EPSG:{epsg}", *geom.bounds, densify_pts=21)
    xmin, ymin, xmax, ymax = bounds
    requested = processing_options()["resolution"]
    resolution = max(requested, math.ceil(max(xmax - xmin, ymax - ymin) / 1800))
    width = max(1, math.ceil((xmax - xmin) / resolution))
    height = max(1, math.ceil((ymax - ymin) / resolution))
    return [xmin, ymax - height * resolution, xmin + width * resolution, ymax], epsg, width, height, resolution


def aoi_pixel_mask(profile, array_shape):
    from rasterio.warp import transform_geom
    geom = transform_geom("EPSG:4326", profile["crs"], ensure_aoi(st.session_state.aoi))
    return geometry_mask([geom], out_shape=array_shape, transform=profile["transform"], invert=True)


def cache_valid(path, count):
    if not path.exists():
        return False
    try:
        with rasterio.open(path) as dataset:
            if dataset.count != count:
                return False
            dataset.read()
        return True
    except Exception:
        return False


def process_download(payload, output_path, count):
    token = get_access_token()
    response = None
    for attempt in range(2):
        try:
            response = requests.post(PROCESS_URL,
                headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
                json=payload, timeout=(30, 180))
        except requests.exceptions.Timeout as exc:
            raise RuntimeError("CDSE download timed out: 30 s connection / 180 s read-inactivity limit. Retry later.") from exc
        except requests.exceptions.RequestException as exc:
            raise RuntimeError("Network error while contacting CDSE. Check the deployment connection.") from exc
        if response.status_code == 401 and attempt == 0:
            auth = st.session_state.get("cdse_auth")
            updated = refresh_cdse_session(auth) if auth else None
            if updated:
                st.session_state.cdse_auth = updated
                token = updated["access_token"]
                continue
        break
    if response.status_code >= 400:
        try:
            error = response.json().get("error", {})
            detail = error.get("message", str(error)) if isinstance(error, dict) else str(error)
        except Exception:
            detail = response.text[:1200]
        raise RuntimeError(f"CDSE Process API failed ({response.status_code}): {detail}")
    try:
        with MemoryFile(response.content) as memory:
            with memory.open() as dataset:
                if dataset.count != count:
                    raise ValueError("Unexpected band count")
                dataset.read()
    except Exception as exc:
        raise RuntimeError("CDSE did not return a readable TIFF with the expected bands.") from exc
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(".tif.part")
    try:
        tmp.write_bytes(response.content)
        tmp.replace(output_path)
    finally:
        tmp.unlink(missing_ok=True)
    return output_path


def day_context(day):
    return {"id": "AOI_DAY_" + day.strftime("%Y%m%d"),
            "properties": {"datetime": day.strftime("%Y-%m-%dT12:00:00Z")}}


def rgb_array(bands):
    rgb = np.stack([bands["B04"], bands["B03"], bands["B02"]], axis=-1)
    rgb = np.power(np.clip(np.nan_to_num(rgb) / 0.30, 0, 1), 0.8)
    return (rgb * 255).astype(np.uint8)


def rgb_png(bands, smoke=None):
    from PIL import Image
    rgb = rgb_array(bands)
    if smoke is not None:
        rgb[smoke] = (0.40 * rgb[smoke] + 0.60 * np.array([230, 57, 70])).astype(np.uint8)
    rgb[bands["dataMask"] < 0.5] = [225, 225, 225]
    buffer = io.BytesIO()
    Image.fromarray(rgb).save(buffer, format="PNG")
    return buffer.getvalue()


def masked_smoke_png(smoke, observed=None, color=(120,120,120)):
    from PIL import Image
    rgb = np.full((*smoke.shape, 3), 255, dtype=np.uint8)
    rgb[smoke.astype(bool)] = color
    if observed is not None:
        rgb[~observed.astype(bool)] = [215,215,215]
    buffer = io.BytesIO()
    Image.fromarray(rgb).save(buffer, format="PNG")
    return buffer.getvalue()


def raster_bytes(array, profile):
    prof = profile.copy()
    prof.update(count=1, dtype="float32", nodata=-9999, compress="deflate", tiled=False)
    with MemoryFile() as memory:
        with memory.open(**prof) as dataset:
            dataset.write(np.where(np.isfinite(array), array, -9999).astype(np.float32), 1)
        return memory.read()


def render_spatial_view(bands, profile, result, aoi, key):
    from rasterio.warp import calculate_default_transform, reproject, Resampling
    from PIL import Image
    rgb = rgb_array(bands)
    alpha = np.where(bands["dataMask"] > 0.5, 255, 0).astype(np.uint8)
    rgba = np.concatenate([rgb, alpha[:,:,None]], axis=2)
    rgba[result["smoke"], :3] = (0.4*rgba[result["smoke"], :3] + 0.6*np.array([230,57,70])).astype(np.uint8)
    h,w = alpha.shape
    bounds = rasterio.transform.array_bounds(h,w,profile["transform"])
    dst_transform,dw,dh = calculate_default_transform(profile["crs"], "EPSG:4326", w,h,*bounds)
    dest = np.zeros((4,dh,dw), dtype=np.uint8)
    for channel in range(4):
        reproject(rgba[:,:,channel], dest[channel], src_transform=profile["transform"],
                  src_crs=profile["crs"], dst_transform=dst_transform, dst_crs="EPSG:4326",
                  resampling=Resampling.nearest)
    west,south,east,north = rasterio.transform.array_bounds(dh,dw,dst_transform)
    geom=shape(ensure_aoi(aoi))
    fmap=folium.Map([geom.centroid.y,geom.centroid.x],tiles="OpenStreetMap")
    folium.raster_layers.ImageOverlay(np.moveaxis(dest,0,-1), bounds=[[south,west],[north,east]],
                                     name="S2 RGB + smoke candidates", opacity=1).add_to(fmap)
    folium.GeoJson(ensure_aoi(aoi), name="AOI", style_function=lambda _: {"color":"#457b9d","fill":False}).add_to(fmap)
    fmap.fit_bounds([[geom.bounds[1],geom.bounds[0]],[geom.bounds[3],geom.bounds[2]]])
    folium.LayerControl().add_to(fmap)
    st.caption("Basemap is geographic context only, not a satellite observation. Transparent gaps contain no S2 observation.")
    st_folium(fmap,height=480,width=1200,key=key,returned_objects=[])


def s2_evalscript(target_day=None):
    target = target_day.strftime("%Y-%m-%dT12:00:00Z") if target_day else "2000-01-01T12:00:00Z"
    script = """//VERSION=3
function setup() {
  return {
    input: [{bands:["B02","B03","B04","B08","B11","B12","SCL","dataMask"],
      units:["REFLECTANCE","REFLECTANCE","REFLECTANCE","REFLECTANCE","REFLECTANCE","REFLECTANCE","DN","DN"]}],
    mosaicking: "TILE",
    output: {bands:9, sampleType:"FLOAT32"}
  };
}
function evaluatePixel(samples, scenes) {
  var best=null, score=1e30, day=-9999;
  var target=Date.parse("__TARGET__");
  for(var i=0;i<samples.length;i++) {
    var s=samples[i];
    if(s.dataMask !== 1 || s.SCL===0 || s.SCL===1) continue;
    var date=scenes.tiles[i].date;
    var stamp=Date.parse(date);
    if(!isFinite(stamp)) continue;
    var cloud=([3,8,9,10,11].indexOf(s.SCL)>=0)?1000:0;
    var rank=cloud+Math.abs(stamp-target)/86400000;
    if(rank<score) {best=s;score=rank;day=Math.floor(stamp/86400000);}
  }
  if(best===null) return [0,0,0,0,0,0,0,0,-9999];
  return [best.B02,best.B03,best.B04,best.B08,best.B11,best.B12,best.SCL,1,day];
}
"""
    return script.replace("__TARGET__", target)


def download_s2_scene(item, aoi, access_token=None):
    aoi = ensure_aoi(aoi)
    acquisition = get_datetime(item)
    if acquisition is None:
        raise RuntimeError("Select a valid observation date first.")
    settings = processing_options()
    bounds, epsg, width, height, resolution = grid_for_aoi(aoi)
    pad = settings["window"]
    start = acquisition.replace(hour=0,minute=0,second=0,microsecond=0) - timedelta(days=pad)
    end = start + timedelta(days=2*pad+1)
    cache_id=hashlib.sha256(json.dumps(["aoi_mosaic_v4",aoi,str(start),str(end),epsg,width,height],sort_keys=True).encode()).hexdigest()[:24]
    output_path=CACHE_DIR / cache_id / "bands.tif"
    if cache_valid(output_path,len(BANDS)):
        return output_path
    payload={
        "input":{"bounds":{"bbox":bounds,"properties":{"crs":f"http://www.opengis.net/def/crs/EPSG/0/{epsg}"}},
                 "data":[{"type":"sentinel-2-l2a","dataFilter":{
                     "timeRange":{"from":start.strftime("%Y-%m-%dT%H:%M:%SZ"),"to":end.strftime("%Y-%m-%dT%H:%M:%SZ")},
                     "mosaickingOrder":"mostRecent","maxCloudCoverage":100},
                     "processing":{"upsampling":"NEAREST","downsampling":"NEAREST"}}]},
        "output":{"width":width,"height":height,"responses":[{"identifier":"default","format":{"type":"image/tiff"}}]},
        "evalscript":s2_evalscript(acquisition)}
    return process_download(payload,output_path,len(BANDS))


def read_s2_stack(path, aoi=None):
    from rasterio.warp import transform_geom
    with rasterio.open(path) as source:
        array=source.read().astype(np.float32)
        profile=source.profile.copy()
    bands={band:array[index] for index,band in enumerate(BANDS)}
    geom=transform_geom("EPSG:4326",profile["crs"],ensure_aoi(aoi if aoi is not None else st.session_state.aoi))
    inside=geometry_mask([geom],out_shape=array.shape[1:],transform=profile["transform"],invert=True)
    observed=inside & (bands["dataMask"]>0.5)
    bands["dataMask"]=observed.astype(np.float32)
    for band in BANDS:
        if band!="dataMask":
            bands[band][~observed]=np.nan
    return bands,profile


def normalized_difference(first, second):
    output = np.full(first.shape, np.nan, dtype=np.float32)
    denominator = first + second
    valid = np.isfinite(first) & np.isfinite(second) & (np.abs(denominator) > 1e-12)
    output[valid] = (first[valid] - second[valid]) / denominator[valid]
    return output


def calculate_smoke_mask(bands, params):
    b02,b03,b04,b08,b11,b12 = [bands[k] for k in BANDS[:6]]
    finite = np.logical_and.reduce([np.isfinite(bands[k]) for k in BANDS[:6]])
    observed = finite & (bands["dataMask"] > 0.5)
    scl = np.nan_to_num(bands["SCL"],nan=0).astype(np.int32)
    rejected = [0,1,3,9,10,11]
    if not processing_options()["keep_medium"]:
        rejected.append(8)
    cloud_shadow = np.isin(scl,rejected)
    snow = scl == 11
    water = scl == 6
    valid = observed & ~cloud_shadow
    smoke_index = normalized_difference(b03,b11)
    ndvi = normalized_difference(b08,b04)
    brightness = (b02+b03+b04)/3.0
    ratio = np.divide(b12,b11,out=np.full_like(b11,np.nan),where=np.isfinite(b11)&(b11>1e-6))
    # Water is not excluded: a plume over the sea must remain observable.
    # This is an unvalidated spectral candidate screen, not a smoke classifier.
    candidate = (valid & (smoke_index > params["smoke_index_threshold"])
        & (brightness > params["blue_brightness_min"])
        & (ratio < params["swir_ratio_max"])
        & (np.nan_to_num(ndvi,nan=1.0) < 0.45))
    if params["gaussian_sigma"] > 0:
        weights=gaussian_filter(valid.astype(np.float32),sigma=params["gaussian_sigma"])
        smooth=gaussian_filter(np.where(valid,smoke_index,0.0),sigma=params["gaussian_sigma"])
        smooth=np.divide(smooth,weights,out=np.full_like(smooth,np.nan),where=weights>0.1)
        candidate &= smooth > params["smoke_index_threshold"]
    labeled,n_labels=label(candidate,structure=np.ones((3,3),dtype=np.uint8))
    sizes=np.bincount(labeled.ravel(),minlength=n_labels+1)
    keep=np.where(sizes >= int(params["min_component_pixels"]))[0]
    keep=keep[keep!=0]
    smoke=np.isin(labeled,keep) & valid
    if smoke.any() and params["final_dilation"]>0:
        smoke=binary_dilation(smoke,structure=disk(int(params["final_dilation"]))) & valid
    fraction=float(smoke.sum()/max(1,valid.sum()))
    diagnostics={
        "smoke_index_max":float(np.nanmax(smoke_index[valid])) if valid.any() else np.nan,
        "smoke_index_mean":float(np.nanmean(smoke_index[valid])) if valid.any() else np.nan,
        "brightness_mean":float(np.nanmean(brightness[valid])) if valid.any() else np.nan,
        "smoke_fraction":fraction,
        "large_mask_warning":fraction>params["max_smoke_fraction"],
        "observed_pixels":int(observed.sum())}
    return {"smoke":smoke,"valid":valid,"observed":observed,"cloud_shadow":cloud_shadow,
            "water":water,"snow":snow,"smoke_index":np.where(observed,smoke_index,np.nan),
            "brightness":brightness,"diagnostics":diagnostics}


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
    script = """//VERSION=3
function setup() {
  return {input:[{bands:["__BAND__","dataMask"]}],
          mosaicking:"ORBIT", output:{bands:3,sampleType:"FLOAT32"}};
}
function evaluatePixel(samples) {
  var total=0,count=0;
  for(var i=0;i<samples.length;i++) {
    var s=samples[i];
    if(s.dataMask===1 && isFinite(s.__BAND__)) {total+=s.__BAND__;count++;}
  }
  return count>0 ? [total/count,1,count] : [0,0,0];
}
"""
    return script.replace("__BAND__",band)


def download_s5p_layer(aoi, date_from, date_to, pollutant, access_token=None):
    config=S5P_POLLUTANTS[pollutant]
    band=config["band"]
    aoi=ensure_aoi(aoi)
    options=processing_options()
    qa=options["qa"] if options["qa_override"] else (75 if pollutant=="NO2" else 50)
    date_from=datetime.combine(date_from.date(),datetime.min.time())
    date_to=datetime.combine(date_to.date(),datetime.min.time())+timedelta(days=1)
    cache_id=hashlib.sha256(json.dumps(["s5p_orbit_v4",band,aoi,str(date_from),str(date_to),qa],sort_keys=True).encode()).hexdigest()[:24]
    output_path=CACHE_DIR / f"s5p_{cache_id}" / "layer.tif"
    if cache_valid(output_path,3):
        return output_path
    minx,miny,maxx,maxy=shape(aoi).bounds
    ratio=max(0.1,(maxx-minx)*math.cos(math.radians((miny+maxy)/2))/max(1e-6,maxy-miny))
    width=min(512,max(1,round(384*min(1,ratio))))
    height=min(512,max(1,round(384*min(1,1/ratio))))
    payload={"input":{"bounds":{"bbox":[minx,miny,maxx,maxy],
        "properties":{"crs":"http://www.opengis.net/def/crs/EPSG/0/4326"}},
        "data":[{"type":"sentinel-5p-l2","dataFilter":{
            "timeRange":{"from":date_from.strftime("%Y-%m-%dT00:00:00Z"),"to":date_to.strftime("%Y-%m-%dT00:00:00Z")},
            "mosaickingOrder":"mostRecent"},"processing":{"minQa":qa,"upsampling":"NEAREST"}}]},
        "output":{"width":width,"height":height,"responses":[{"identifier":"default","format":{"type":"image/tiff"}}]},
        "evalscript":s5p_evalscript(band)}
    return process_download(payload,output_path,3)


def read_s5p(path):
    with rasterio.open(path) as source:
        values=source.read(1).astype(np.float32)
        mask=source.read(2)>0.5
        profile=source.profile.copy()
    mask &= aoi_pixel_mask(profile,values.shape)
    values[~mask | ~np.isfinite(values)] = np.nan
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
    return masked_smoke_png(np.asarray(array).astype(bool))


def pollutant_png(array, limits=None):
    from PIL import Image
    import matplotlib.pyplot as plt
    data=np.asarray(array,dtype=np.float32)
    finite=np.isfinite(data)
    rgb=np.full((*data.shape,3),225,dtype=np.uint8)
    if finite.any():
        low,high=limits if limits is not None else np.percentile(data[finite],[2,98])
        if not np.isfinite(low) or not np.isfinite(high):
            low,high=0.0,1.0
        if high<=low:
            span=max(abs(float(low))*0.05,1e-9)
            low,high=float(low)-span,float(high)+span
        normalized=np.clip((np.nan_to_num(data,nan=low)-low)/(high-low),0,1)
        rgb=(plt.get_cmap("inferno")(normalized)[:,:,:3]*255).astype(np.uint8)
        rgb[~finite]=[225,225,225]
    buffer=io.BytesIO()
    Image.fromarray(rgb).save(buffer,format="PNG")
    return buffer.getvalue()


def legend_html(kind):
    if kind == "smoke":
        rows = [("#787878", "Smoke candidate"), ("#ffffff", "No candidate"), ("#d7d7d7", "No observation")]
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


def georeferenced_png_package(array, profile, mask=False, observed=None, color=(120,120,120)):
    png_data=masked_smoke_png(array,observed,color) if mask else pollutant_png(array)
    t=profile["transform"]
    pgw=f"{t.a:.12f}\n{t.d:.12f}\n{t.b:.12f}\n{t.e:.12f}\n{t.c+t.a/2+t.b/2:.12f}\n{t.f+t.d/2+t.e/2:.12f}\n"
    crs_text=profile.get("crs")
    package=io.BytesIO()
    with zipfile.ZipFile(package,"w",zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("output.png",png_data)
        archive.writestr("output.pgw",pgw)
        if crs_text:
            archive.writestr("output.prj",crs_text.to_wkt())
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
    <div class="status-pill">AOI-wide S2 mosaics &nbsp;&nbsp; Multi-pollutant S5P</div>
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
        max_cloud = st.slider("Cloud metadata warning (%)", 0.0, 100.0, 40.0, key="max_cloud")
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
                st.success(f"{len(search_results)} tile record(s) found")
                cloudy_records=sum(get_cloud(item)>max_cloud for item in search_results)
                st.caption(f"{cloudy_records} tile record(s) exceed the cloud metadata warning threshold; they remain available for coverage.")
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

        available_days = sorted({get_datetime(s).strftime("%Y-%m-%d") for s in scene_results if get_datetime(s)})
        st.session_state.available_days = available_days
        if available_days:
            selected_day = st.selectbox("Observation date (all available tiles in AOI)", available_days,
                                        key="target_day_select")
            st.session_state["target"] = day_context(datetime.fromisoformat(selected_day))
        st.caption("No reference image is used. The date selects all available S2 tiles for your AOI. Cloud metadata does not remove tiles from the mosaic.")

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
            min_value=-0.50, max_value=0.80,
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


    with st.expander("AOI mosaic, quality and regional processing", expanded=False):
        st.selectbox("Requested S2 grid spacing (m)", [20,60,100,250,500,1000], index=3,
                     key="regional_resolution")
        st.selectbox("S2 composite half-window (days)", [0,1,3,7], key="s2_composite_days",
                     help="0 = selected date only. Other values mix observations from different dates, explicitly labelled.")
        st.checkbox("Include SCL medium-cloud pixels (experimental; higher false-positive risk)",
                    value=False,key="keep_medium_cloud")
        SMOKE_PARAMS["blue_brightness_min"] = st.slider("Minimum visible brightness",0.01,0.40,0.10,0.01,key="minimum_brightness")
        SMOKE_PARAMS["swir_ratio_max"] = st.slider("Maximum B12/B11 ratio",0.20,3.00,1.60,0.05,key="maximum_swir_ratio")
        st.checkbox("Override S5P official default QA thresholds",value=False,key="s5p_qa_override")
        st.slider("S5P minimum QA (%) when override enabled",0,100,50,key="s5p_min_qa")
        st.caption("Lower QA is exploratory, not a validated replacement for official quality filtering. Zero and negative valid S5P retrievals are retained.")
    grid_bounds, grid_epsg, grid_w, grid_h, grid_res = grid_for_aoi(st.session_state.aoi)
    st.caption(f"Actual regional grid: {grid_res} m | {grid_w} × {grid_h} | EPSG:{grid_epsg}. "
               "Long side capped at 1800 pixels to bound memory use; large AOIs are not processed at native 20 m.")
    estimated_area_m2 = int(SMOKE_PARAMS["min_component_pixels"]) * grid_res * grid_res
    st.markdown(
        f'<div class="card-caption">Minimum connected smoke region ≈ {estimated_area_m2:,} m² at {grid_res} m processing-grid spacing.</div>',
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
            target_tile = "All available AOI tiles"
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
                        '<div class="card-caption">Step 1 of 4 · Downloading all available AOI tiles for the date/window…</div>',
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
                    output_folder = RESULT_DIR / (target_date.strftime("%Y%m%d") + "_" + s2_path.parent.name)
                    output_folder.mkdir(parents=True, exist_ok=True)
                    paths = {}
                    for key in ("smoke", "valid", "smoke_index"):
                        paths[key] = output_folder / f"{key}.tif"
                        if key in ("smoke","valid"):
                            known=result["valid"] if key=="smoke" else result["observed"]
                            output=np.where(known,result[key].astype(np.float32),np.nan)
                            save_raster(paths[key],output,profile,mask=False)
                        else:
                            save_raster(paths[key],result[key],profile,mask=False)

                    inside = aoi_pixel_mask(profile,result["valid"].shape)
                    st.session_state.smoke_context = {
                        "date":target_date.strftime("%Y-%m-%d"),
                        "window":processing_options()["window"],
                        "coverage":float(result["observed"].sum()/max(1,inside.sum())),
                        "valid_fraction":float(result["valid"].sum()/max(1,inside.sum())),
                        "resolution":abs(profile["transform"].a),
                        "aoi":ensure_aoi(st.session_state.aoi),
                        "params":dict(SMOKE_PARAMS),
                        "bands_path":str(s2_path),
                    }
                    save_raster(output_folder / "observation_day.tif", bands["observation_day"], profile)
                    st.session_state.smoke_result = result
                    st.session_state.smoke_paths = paths
                    st.session_state.smoke_profile = profile
                    st.session_state.smoke_pngs = {
                        "smoke": masked_smoke_png(result["smoke"],result["observed"]),
                        "valid": masked_smoke_png(result["valid"],result["observed"],color=(220,30,30)),
                        "smoke_index": pollutant_png(result["smoke_index"]),
                    }

                    progress_status.markdown(
                        '<div class="card-caption">Step 4 of 4 · Preparing downloads…</div>',
                        unsafe_allow_html=True,
                    )
                    progress.progress(100, text="Smoke detection completed")
                    st.success("Smoke detection completed")

                    if not result["valid"].any():
                        st.warning("No assessable S2 pixels. This does not mean no smoke.")
                    elif result["smoke"].sum() == 0:
                        st.warning("No candidate under this heuristic. This is not evidence of smoke absence.")
                    if result["diagnostics"]["large_mask_warning"]:
                        st.warning("Very large candidate fraction: possible water, haze or cloud contamination. Mask retained for inspection.")
                except Exception as error:
                    st.error(f"Smoke detection failed: {error}")
                finally:
                    progress.empty()
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

    context = st.session_state.get("smoke_context",{})
    if context:
        st.caption(f"Date: {context['date']} | composite half-window: ±{context['window']} day(s) | actual grid: {context['resolution']:.0f} m")
        c1,c2,c3=st.columns(3)
        c1.metric("Observed AOI coverage",f"{context['coverage']*100:.1f}%")
        c2.metric("Assessable AOI coverage",f"{context['valid_fraction']*100:.1f}%")
        c3.metric("Candidate area (km²)",f"{result['smoke'].sum()*context['resolution']**2/1e6:.2f}")
        if context['coverage']<0.99:
            st.warning("Incomplete observation coverage: no-data pixels are unknown, not clean. Try another date or an explicitly multi-date composite.")
        if ensure_aoi(st.session_state.aoi)!=context["aoi"]:
            st.warning("AOI changed since this result. Run again before interpreting these outputs.")
        if context["window"]:
            st.info("Multi-date mosaic: pixels may come from different days. Not a single-time observation; use the observation-day raster to inspect dates.")
        b_preview,p_preview=read_s2_stack(Path(context["bands_path"]),context["aoi"])
        left,right=st.columns(2)
        with left:
            st.caption("Sentinel-2 RGB — grey means no observation")
            st.image(rgb_png(b_preview),use_container_width=True)
        with right:
            st.caption("RGB + candidate overlay — red is a spectral candidate, not confirmed smoke")
            st.image(rgb_png(b_preview,result["smoke"]),use_container_width=True)
        with st.expander("Geographic RGB + candidate overlay",expanded=False):
            render_spatial_view(b_preview,p_preview,result,context["aoi"],"result_geographic_map")
        st.download_button("⬇ Observation-date raster (days since 1970-01-01)",
            raster_bytes(b_preview["observation_day"],p_preview),file_name="observation_day.tif",mime="image/tiff",key="observation_day_download")
    st.warning("Exploratory spectral screening only: clouds, haze, bright water and other surfaces may be confused with smoke; dark smoke may be missed. No conflict-source attribution is inferred.")
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
                    observed=result["observed"],
                    color=(220,30,30) if key=="valid" else (120,120,120),
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

    saved_smoke_params=st.session_state.get("smoke_context",{}).get("params",SMOKE_PARAMS)
    st.markdown(
        f'<div class="result-note"><b>Smoke detection:</b> smoke index = (B03 − B11) / (B03 + B11). '
        f'Cloud/shadow/snow quality flags are applied; water is retained to allow marine smoke candidates. '
        f'Initial threshold = <b>{saved_smoke_params["smoke_index_threshold"]:.2f}</b>; '
        f'minimum connected pixels = <b>{int(saved_smoke_params["min_component_pixels"])}</b>; '
        f'final dilation = <b>{int(saved_smoke_params["final_dilation"])}</b>.</div>',
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
    s5p_progress=None
    try:
        access_token = get_access_token()
        target_date = get_datetime(st.session_state.get("target"))
        if target_date is None:
            raise RuntimeError("Target date is missing.")

        date_from = target_date - timedelta(days=int(s5p_window))
        date_to = target_date + timedelta(days=int(s5p_window))

        s5p_data = {}
        s5p_paths = {}
        s5p_errors=[]
        s5p_progress = st.progress(0, text="Fetching Sentinel-5P layers…")
        for index, pollutant in enumerate(selected_pollutants, start=1):
            s5p_progress.progress(int(100*(index-1)/len(selected_pollutants)),
                text=f"Fetching {S5P_POLLUTANTS[pollutant]['label']}…")
            try:
                path=download_s5p_layer(st.session_state.aoi,date_from,date_to,pollutant,access_token)
                s5p_data[pollutant]=read_s5p(path)
                s5p_paths[pollutant]=str(path)
            except Exception as exc:
                s5p_errors.append({"product":pollutant,"error":str(exc)})
        st.session_state.s5p_errors=s5p_errors
        st.session_state.s5p_data = s5p_data
        st.session_state.s5p_paths = s5p_paths
        st.session_state.s5p_aoi = ensure_aoi(st.session_state.aoi)
        st.session_state.s5p_date_range = (date_from, date_to)
        s5p_progress.progress(100, text="Sentinel-5P layers ready")
        valid_layers=sum(np.isfinite(v).any() for v in s5p_data.values())
        if valid_layers:
            st.success(f"S5P returned valid observations for {valid_layers}/{len(s5p_data)} layer(s).")
        else:
            st.warning("Requests completed, but no valid S5P observations. This is not zero pollution. Inspect dates, QA and acquisition availability.")
    except Exception as s5p_error:
        st.error(f"S5P fetch failed: {s5p_error}")
    finally:
        if s5p_progress is not None:
            s5p_progress.empty()

if "s5p_data" in st.session_state:
    s5p_data = st.session_state.s5p_data
    if st.session_state.get("s5p_errors"):
        with st.expander("S5P product request errors",expanded=True):
            st.dataframe(pd.DataFrame(st.session_state.s5p_errors),use_container_width=True)
    dr=st.session_state.get("s5p_date_range")
    if dr:
        st.caption(f"S5P observation mean: {dr[0].date()} to {dr[1].date()} inclusive — multi-date, not one snapshot.")
    st.caption("QA defaults: NO2 75%, other products 50% unless explicitly overridden. Grey = no valid retrieval. Valid zero/negative values are not discarded.")
    if st.session_state.get("s5p_aoi")!=ensure_aoi(st.session_state.aoi):
        st.warning("AOI changed since these S5P results. Fetch again.")
    pollutant_cols = st.columns(min(3, len(s5p_data)) if s5p_data else 1, gap="small")

    for index, (pollutant, values) in enumerate(s5p_data.items()):
        with pollutant_cols[index % len(pollutant_cols)]:
            config = S5P_POLLUTANTS[pollutant]
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.markdown(f'<div class="result-tag">{config["label"]}</div>', unsafe_allow_html=True)
            valid_values = values[np.isfinite(values)]
            layer_path=st.session_state.get("s5p_paths",{}).get(pollutant)
            if layer_path:
                with rasterio.open(layer_path) as ds:
                    layer_profile=ds.profile.copy()
                from rasterio.warp import transform_geom
                saved_geom=transform_geom("EPSG:4326",layer_profile["crs"],st.session_state.get("s5p_aoi",ensure_aoi(st.session_state.aoi)))
                inside=geometry_mask([saved_geom],out_shape=values.shape,transform=layer_profile["transform"],invert=True)
                st.caption(f"Valid retrieval coverage: {np.isfinite(values).sum()/max(1,inside.sum())*100:.1f}% of AOI display cells")
                st.download_button("⬇ S5P GeoTIFF",raster_bytes(values,layer_profile),
                    file_name=f"{pollutant}.tif",mime="image/tiff",key=f"s5p_tif_{pollutant}")
            if valid_values.size:
                st.metric(
                    f"Mean {pollutant}",
                    f"{float(np.nanmean(valid_values)):.3g} {config['unit']}",
                )
                st.image(pollutant_png(values), use_container_width=True, output_format="PNG")
            else:
                st.warning("No valid retrievals in this window. Unknown, not zero pollution; inspect dates, QA and coverage.")
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
            series_errors=[]
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
                    except Exception as exc:
                        row[pollutant] = np.nan
                        series_errors.append({"date":str(day.date()),"product":pollutant,"error":str(exc)})
                    step += 1
                    ts_progress.progress(
                        int(100 * step / total_steps),
                        text=f"{day.date()} · {pollutant}",
                    )
                rows.append(row)

            ts_df = pd.DataFrame(rows).set_index("date")
            st.session_state.s5p_timeseries = ts_df
            st.session_state.s5p_series_errors=series_errors
            ts_progress.progress(100, text="S5P time series ready")
            if ts_df.notna().any().any():
                st.success("S5P time series built; gaps are missing observations, not zeros.")
            else:
                st.warning("No valid S5P observations in this series. Missing/error days remain NaN.")
        except Exception as ts_error:
            st.error(f"S5P time series failed: {ts_error}")

    if "s5p_timeseries" in st.session_state:
        ts = st.session_state.s5p_timeseries
        if st.session_state.get("s5p_series_errors"):
            with st.expander("Standalone S5P series request errors"):
                st.dataframe(pd.DataFrame(st.session_state.s5p_series_errors),use_container_width=True)
        if not ts.empty:
            st.markdown(
                '<div class="card-caption">Daily mean values over the AOI for selected Sentinel-5P pollutants</div>',
                unsafe_allow_html=True,
            )
            for pollutant in ts.columns:
                st.caption(f"{S5P_POLLUTANTS[pollutant]['label']} ({S5P_POLLUTANTS[pollutant]['unit']})")
                st.line_chart(ts[[pollutant]],use_container_width=True,height=180)
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


# Monthly synchronized playback: daily observations, no reference subtraction.
st.markdown('<div style="height:0.35rem"></div>',unsafe_allow_html=True)
st.markdown('<div class="app-card">',unsafe_allow_html=True)
st.markdown('<div class="section-label">06 · MONTHLY S2 + S5P SYNCHRONIZED PLAYBACK</div>',unsafe_allow_html=True)
st.markdown('<div class="card-title">Daily changes across the selected AOI</div>',unsafe_allow_html=True)
st.caption("Uses the search start/end dates above. Each calendar day is represented; missing observations are never interpolated or replaced with a clean image.")
include_month_s5p=st.checkbox("Also download daily Sentinel-5P layers for synchronized playback",value=True,key="include_month_s5p")
st.caption("A month with four pollutants can require more than 100 API requests. First build may be slow; successfully downloaded rasters are cached. The S5P window above is not used here: these are daily retrievals.")
if st.button("🎬 Build monthly S2 + S5P playback",type="primary",use_container_width=True,
             key="build_joint_month",disabled=not bool(st.session_state.get("cdse_auth"))):
    monthly_progress=st.progress(0,text="Preparing monthly observations…")
    monthly_status=st.empty()
    try:
        if end_date<start_date:
            raise RuntimeError("Start date must be before end date.")
        if (end_date-start_date).days>61:
            raise RuntimeError("Build at most 62 calendar days at a time to keep the hosted app responsive.")
        aoi=ensure_aoi(st.session_state.aoi)
        days=pd.date_range(start_date,end_date,freq="D")
        discovery=search_scenes(aoi,datetime.combine(start_date,datetime.min.time())-timedelta(days=processing_options()["window"]),
            datetime.combine(end_date,datetime.max.time())+timedelta(days=processing_options()["window"]),100)
        observed_days={get_datetime(s).date() for s in discovery if get_datetime(s)}
        pollutants=list(selected_pollutants) if include_month_s5p else []
        settings=processing_options()
        signature=hashlib.sha256(json.dumps([aoi,settings,SMOKE_PARAMS,str(start_date),str(end_date),pollutants],sort_keys=True).encode()).hexdigest()[:20]
        frame_dir=RESULT_DIR / ("monthly_"+signature)
        frame_dir.mkdir(parents=True,exist_ok=True)
        rows=[]
        frames={}
        errors=[]
        sample_values={p:[] for p in pollutants}
        for index,day in enumerate(days):
            date_key=day.strftime("%Y-%m-%d")
            monthly_status.caption(f"{date_key} — S2 and selected S5P products")
            frame={"date":date_key,"s5p":{},"s2_status":"no acquisition"}
            row={"date":day,"s2_observed_pct":np.nan,"s2_assessable_pct":np.nan,
                 "smoke_candidate_km2":np.nan,"smoke_candidate_fraction":np.nan,"s2_status":"no acquisition"}
            pad=settings["window"]
            potential=any(abs((d-day.date()).days)<=pad for d in observed_days)
            if potential:
                try:
                    path=download_s2_scene(day_context(day.to_pydatetime()),aoi)
                    bands,profile=read_s2_stack(path,aoi)
                    result=calculate_smoke_mask(bands,SMOKE_PARAMS)
                    inside=aoi_pixel_mask(profile,result["smoke"].shape)
                    observed_fraction=result["observed"].sum()/max(1,inside.sum())
                    assessable_fraction=result["valid"].sum()/max(1,inside.sum())
                    status="observed" if result["observed"].any() else "no observation"
                    if result["observed"].any() and not result["valid"].any():
                        status="observed but quality-masked"
                    frame.update({"s2_status":status,"s2_path":str(path),"resolution":abs(profile["transform"].a)})
                    rgb_path=frame_dir/(date_key+"_rgb.png")
                    overlay_path=frame_dir/(date_key+"_overlay.png")
                    mask_path=frame_dir/(date_key+"_mask.png")
                    rgb_path.write_bytes(rgb_png(bands))
                    overlay_path.write_bytes(rgb_png(bands,result["smoke"]))
                    mask_path.write_bytes(masked_smoke_png(result["smoke"],result["observed"]))
                    frame.update({"rgb_png":str(rgb_path),"overlay_png":str(overlay_path),"mask_png":str(mask_path)})
                    mask_tif=frame_dir/(date_key+"_smoke.tif")
                    candidate=np.where(result["valid"],result["smoke"].astype(np.float32),np.nan)
                    save_raster(mask_tif,candidate,profile)
                    frame["mask_tif"]=str(mask_tif)
                    area=float(result["smoke"].sum()*abs(profile["transform"].a*profile["transform"].e)/1e6)
                    row.update({"s2_observed_pct":observed_fraction*100,"s2_assessable_pct":assessable_fraction*100,
                        "smoke_candidate_km2":area if result["valid"].any() else np.nan,
                        "smoke_candidate_fraction":result["diagnostics"]["smoke_fraction"] if result["valid"].any() else np.nan,
                        "s2_status":status})
                    del bands,result
                except Exception as exc:
                    frame["s2_status"]="request failed"
                    frame["s2_error"]=str(exc)
                    row["s2_status"]="request failed"
                    errors.append({"date":date_key,"sensor":"S2","error":str(exc)})
            else:
                row["s2_observed_pct"]=0.0
                row["s2_assessable_pct"]=0.0
            for pollutant in pollutants:
                row[pollutant]=np.nan
                row[pollutant+"_coverage_pct"]=np.nan
                try:
                    path=download_s5p_layer(aoi,day,day,pollutant)
                    values=read_s5p(path)
                    finite=values[np.isfinite(values)]
                    with rasterio.open(path) as ds:
                        inside=aoi_pixel_mask(ds.profile,values.shape)
                    coverage=float(finite.size/max(1,inside.sum())*100)
                    row[pollutant+"_coverage_pct"]=coverage
                    row[pollutant]=float(finite.mean()) if finite.size else np.nan
                    frame["s5p"][pollutant]={"path":str(path),"coverage":coverage,
                        "mean":row[pollutant],"status":"valid" if finite.size else "no valid retrieval"}
                    if finite.size:
                        sample_values[pollutant].append(finite[::max(1,finite.size//2000)])
                except Exception as exc:
                    frame["s5p"][pollutant]={"status":"request failed","error":str(exc)}
                    errors.append({"date":date_key,"sensor":pollutant,"error":str(exc)})
            frames[date_key]=frame
            rows.append(row)
            monthly_progress.progress(int(100*(index+1)/len(days)),text=f"Processed {index+1}/{len(days)} calendar days")
        limits={}
        for pollutant,chunks in sample_values.items():
            if chunks:
                pooled=np.concatenate(chunks)
                low,high=np.percentile(pooled,[2,98])
                if high<=low:
                    span=max(abs(float(low))*0.05,1e-9)
                    low,high=low-span,high+span
                limits[pollutant]=(float(low),float(high))
        df=pd.DataFrame(rows).set_index("date")
        st.session_state.joint_month={"frames":frames,"df":df,"errors":errors,"aoi":aoi,
            "settings":settings,"params":dict(SMOKE_PARAMS),"limits":limits,"pollutants":pollutants,
            "start":str(start_date),"end":str(end_date),"signature":signature}
        st.success(f"Playback ready for {len(frames)} calendar days. Missing observations and errors are labelled.")
    except Exception as exc:
        st.error(f"Monthly build failed: {exc}")
    finally:
        monthly_progress.empty()
        monthly_status.empty()

if "joint_month" in st.session_state:
    month=st.session_state.joint_month
    if month["aoi"]!=ensure_aoi(st.session_state.aoi) or month["settings"]!=processing_options() or month["params"]!=dict(SMOKE_PARAMS) or month["start"]!=str(start_date) or month["end"]!=str(end_date):
        st.warning("AOI/settings have changed. The playback below still belongs to its saved AOI/settings; rebuild to update it.")
    dates=sorted(month["frames"])
    st.caption(f"Saved interval: {month['start']} to {month['end']} | S2 composite half-window: ±{month['settings']['window']} day(s)")
    slider_key="joint_slider_"+month["signature"]
    selected_date=st.select_slider("Shared date — Sentinel-2 and Sentinel-5P",options=dates,value=dates[0],key=slider_key)
    frame=month["frames"][selected_date]
    row=month["df"].loc[pd.Timestamp(selected_date)]
    st.markdown(f"### {selected_date}")
    st.caption("Same requested date, not simultaneous overpasses. S5P and S2 have different spatial support and acquisition times.")
    if month["settings"]["window"]:
        st.info("S2 uses mixed dates in the stated window. Successive slider frames may reuse observations; do not interpret this as independent daily change.")
    if "rgb_png" in frame:
        left,right=st.columns(2)
        with left:
            st.caption("S2 RGB — full saved AOI grid")
            st.image(Path(frame["rgb_png"]).read_bytes(),use_container_width=True)
        with right:
            st.caption("S2 RGB + red smoke candidates")
            st.image(Path(frame["overlay_png"]).read_bytes(),use_container_width=True)
        a,b,c=st.columns(3)
        a.metric("S2 observed coverage",f"{row['s2_observed_pct']:.1f}%")
        b.metric("S2 assessable coverage",f"{row['s2_assessable_pct']:.1f}%")
        c.metric("Candidate area (km²)",f"{row['smoke_candidate_km2']:.2f}" if np.isfinite(row['smoke_candidate_km2']) else "No assessable data")
        st.caption(f"Actual S2 grid: {frame['resolution']:.0f} m. Grey = no observation. No-data and quality-masked areas must not be counted as smoke-free.")
        st.download_button("⬇ This day's smoke-candidate GeoTIFF",Path(frame["mask_tif"]).read_bytes(),
            file_name=selected_date+"_smoke_candidates.tif",mime="image/tiff",key="joint_mask_download")
    else:
        st.warning(f"S2: {frame['s2_status']}. No image has been fabricated for this date.")
        if frame.get("s2_error"):
            st.caption(frame["s2_error"])
    if frame["s5p"]:
        st.markdown("### Sentinel-5P — same calendar day")
        cols=st.columns(min(3,len(frame["s5p"])))
        for i,(pollutant,layer) in enumerate(frame["s5p"].items()):
            with cols[i%len(cols)]:
                config=S5P_POLLUTANTS[pollutant]
                st.caption(config["label"])
                if layer["status"]=="valid":
                    # This path already represents the saved AOI. Apply its saved geometry.
                    with rasterio.open(layer["path"]) as ds:
                        values=ds.read(1).astype(np.float32)
                        good=ds.read(2)>0.5
                        prof=ds.profile.copy()
                    from rasterio.warp import transform_geom
                    geom=transform_geom("EPSG:4326",prof["crs"],month["aoi"])
                    good &= geometry_mask([geom],out_shape=values.shape,transform=prof["transform"],invert=True)
                    values[~good|~np.isfinite(values)]=np.nan
                    st.metric("AOI display-grid mean",f"{layer['mean']:.3g} {config['unit']}")
                    scale=month["limits"].get(pollutant)
                    st.image(pollutant_png(values,scale),use_container_width=True)
                    st.caption(f"Valid coverage: {layer['coverage']:.1f}%")
                    if scale:
                        st.caption(f"Fixed monthly scale: {scale[0]:.3g} to {scale[1]:.3g} {config['unit']}")
                else:
                    st.warning(layer["status"]+" — unknown, not zero")
                    if layer.get("error"):
                        st.caption(layer["error"])
    st.markdown("### Monthly curves and coverage")
    st.caption("Candidate area is over observed/assessable pixels only. Check coverage before comparing days; gaps are not filled.")
    st.line_chart(month["df"][["smoke_candidate_km2"]],height=200)
    st.line_chart(month["df"][["s2_observed_pct","s2_assessable_pct"]],height=180)
    for pollutant in month["pollutants"]:
        st.caption(f"{S5P_POLLUTANTS[pollutant]['label']} ({S5P_POLLUTANTS[pollutant]['unit']}) — daily display-grid mean")
        st.line_chart(month["df"][[pollutant]],height=180)
    st.dataframe(month["df"],use_container_width=True)
    st.download_button("⬇ Monthly observations and coverage CSV",month["df"].to_csv(),
        file_name="monthly_observations_coverage.csv",mime="text/csv",key="joint_csv")
    if st.button("📦 Prepare monthly PNG/CSV/GeoTIFF package",key="joint_prepare_zip"):
        package_path=RESULT_DIR/("monthly_"+month["signature"]+".zip")
        with zipfile.ZipFile(package_path,"w",zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("observations_coverage.csv",month["df"].to_csv())
            archive.writestr("metadata.json",json.dumps({k:v for k,v in month.items() if k not in ("frames","df")},ensure_ascii=False,default=str,indent=2))
            for date_key,frame_record in month["frames"].items():
                for asset in ("rgb_png","overlay_png","mask_png","mask_tif"):
                    if asset in frame_record:
                        asset_path=Path(frame_record[asset])
                        archive.write(asset_path,arcname=date_key+"/"+asset_path.name)
                for product,record in frame_record["s5p"].items():
                    if record.get("path"):
                        archive.write(record["path"],arcname=date_key+"/S5P_"+product+".tif")
        st.session_state.joint_package=str(package_path)
    package_path=st.session_state.get("joint_package")
    if package_path and Path(package_path).name=="monthly_"+month["signature"]+".zip" and Path(package_path).exists():
        st.download_button("⬇ Download monthly package",Path(package_path).read_bytes(),file_name=Path(package_path).name,mime="application/zip",key="joint_zip_download")
    if month["errors"]:
        with st.expander(f"Request errors ({len(month['errors'])})",expanded=False):
            err_df=pd.DataFrame(month["errors"])
            st.dataframe(err_df,use_container_width=True)
            st.download_button("⬇ Error log CSV",err_df.to_csv(index=False),file_name="request_errors.csv",mime="text/csv",key="joint_errors")
    st.caption("Spectral candidate screening is not a validated smoke map and cannot attribute emissions to a conflict or source. No-data pixels are never filled to simulate complete coverage.")
st.markdown('</div>',unsafe_allow_html=True)
