"""AOI-first Sentinel smoke / methane dashboard.
No reference-image selector. Daily S5P retrievals; automatic past-only S2 context.
S2 composite pixels carry their acquisition date. Stale candidates are not daily smoke.
"""
from __future__ import annotations
import io, json, math, re, hashlib, time, zipfile, base64
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Optional
import traceback
from pyproj import Geod
import numpy as np
import pandas as pd
import requests
import rasterio
from rasterio.io import MemoryFile
from rasterio.features import geometry_mask
from rasterio.warp import transform_bounds, transform_geom, calculate_default_transform, reproject, Resampling
from rasterio.transform import from_bounds
import streamlit as st
import folium
from folium.plugins import Draw, MousePosition
from streamlit_folium import st_folium
from shapely.geometry import box, mapping, shape
from shapely.ops import unary_union
from scipy.ndimage import gaussian_filter, binary_dilation, label
from skimage.morphology import disk
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STAC_URL="https://stac.dataspace.copernicus.eu/v1/"
CATALOG_URL="https://sh.dataspace.copernicus.eu/catalog/v1/search"
TOKEN_URL="https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
PROCESS_URL="https://sh.dataspace.copernicus.eu/api/v1/process"
CACHE_DIR=Path.home()/".smoke_dashboard_v10_available"
RESULT_DIR=CACHE_DIR/"sessions"
CACHE_DIR.mkdir(parents=True,exist_ok=True)
RESULT_DIR.mkdir(parents=True,exist_ok=True)
DEFAULT_AOI=box(51.0,25.0,58.0,29.0)
BANDS=["B02","B03","B04","B08","B11","B12","SCL","dataMask","observation_day"]
EPOCH=date(1970,1,1)
PRODUCTS={
 "CH4":{"band":"CH4","label":"Methane · CH4","unit":"ppb","cmap":"magma","range":[1600,2100]},
 "CO":{"band":"CO","label":"Carbon monoxide · CO","unit":"mol/m²","cmap":"inferno","range":[0,0.1]},
 "NO2":{"band":"NO2","label":"Nitrogen dioxide · NO₂","unit":"mol/m²","cmap":"inferno","range":[0,0.0003]},
 "SO2":{"band":"SO2","label":"Sulfur dioxide · SO₂","unit":"mol/m²","cmap":"inferno","range":[-0.001,0.01]},
 "HCHO":{"band":"HCHO","label":"Formaldehyde · HCHO","unit":"mol/m²","cmap":"inferno","range":[0,0.001]},
 "O3":{"band":"O3","label":"Ozone · O₃","unit":"mol/m²","cmap":"viridis","range":[0.2,0.4]},
 "AI340":{"band":"AER_AI_340_380","label":"UV aerosol index · 340–380","unit":"index","cmap":"plasma","range":[-1,5]},
 "AI354":{"band":"AER_AI_354_388","label":"UV aerosol index · 354–388","unit":"index","cmap":"plasma","range":[-1,5]},
}
CLOUD_PRODUCTS={
 "CLOUD_FRACTION":{"band":"CLOUD_FRACTION","label":"Cloud fraction · context","unit":"fraction","cmap":"Blues","range":[0,1]},
 "CLOUD_OPTICAL_THICKNESS":{"band":"CLOUD_OPTICAL_THICKNESS","label":"Cloud optical thickness · context","unit":"index","cmap":"Blues","range":[0,100]},
 "CLOUD_BASE_HEIGHT":{"band":"CLOUD_BASE_HEIGHT","label":"Cloud base height · context","unit":"m","cmap":"viridis","range":[0,15000]},
 "CLOUD_TOP_HEIGHT":{"band":"CLOUD_TOP_HEIGHT","label":"Cloud top height · context","unit":"m","cmap":"viridis","range":[0,20000]},
 "CLOUD_BASE_PRESSURE":{"band":"CLOUD_BASE_PRESSURE","label":"Cloud base pressure · context","unit":"Pa","cmap":"viridis","range":[10000,110000]},
 "CLOUD_TOP_PRESSURE":{"band":"CLOUD_TOP_PRESSURE","label":"Cloud top pressure · context","unit":"Pa","cmap":"viridis","range":[10000,110000]},
}
ALL_PRODUCTS={**PRODUCTS,**CLOUD_PRODUCTS}

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

def create_map(aoi, search_center=None, search_zoom=6):
    geometry = shape(ensure_aoi(aoi))
    center = (
        [float(search_center[0]), float(search_center[1])]
        if search_center
        else [geometry.centroid.y, geometry.centroid.x]
    )
    zoom = min(int(search_zoom), 7) if search_center else 6

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

    from shapely.geometry import Point
    if (not search_center) or geometry.contains(Point(float(search_center[1]), float(search_center[0]))):
        west, south, east, north = geometry.bounds
        fmap.fit_bounds([[south, west], [north, east]], padding=(15, 15))

    return fmap


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,default=str).encode()).hexdigest()[:24]


def atomic_json(path,value):
    temp=path.with_suffix(".json.part")
    temp.write_text(json.dumps(value,default=str,ensure_ascii=False),encoding="utf-8")
    temp.replace(path)


def clip_mask(profile,h,w,aoi):
    geom=transform_geom("EPSG:4326",profile["crs"],ensure_aoi(aoi))
    return geometry_mask([geom],out_shape=(h,w),transform=profile["transform"],invert=True)


def grid_for(aoi,resolution,max_side=1400):
    west,south,east,north=shape(ensure_aoi(aoi)).bounds
    mid=(south+north)/2
    width_m=(east-west)*111320*math.cos(math.radians(mid))
    height_m=(north-south)*111320
    res=max(resolution,math.ceil(max(width_m,height_m)/max_side))
    return {"bounds":[west,south,east,north],"epsg":4326,
            "w":max(1,math.ceil(width_m/res)),"h":max(1,math.ceil(height_m/res)),"res":res}


def response_error(response):
    try:
        e=response.json().get("error",{})
        return e.get("message",str(e)) if isinstance(e,dict) else str(e)
    except Exception:
        return response.text[:1200]


def process_request(payload,key,bands,force=False):
    path=CACHE_DIR/(key+".tif")
    if path.exists() and not force:
        try:
            with rasterio.open(path) as ds:
                if ds.count==bands:
                    ds.read()
                    return path
        except Exception:
            pass
    token=get_access_token()
    for attempt in range(3):
        try:
            response=requests.post(PROCESS_URL,json=payload,
                headers={"Authorization":f"Bearer {token}","Content-Type":"application/json"},timeout=(25,150))
        except requests.exceptions.Timeout as exc:
            raise RuntimeError("CDSE timeout (25 s connect / 150 s read inactivity); retry resumes from the cache.") from exc
        except requests.exceptions.RequestException as exc:
            raise RuntimeError("Network connection to CDSE failed; retry resumes from the cache.") from exc
        if response.status_code==401 and attempt<2:
            auth=st.session_state.get("cdse_auth")
            updated=refresh_cdse_session(auth) if auth else None
            if updated:
                st.session_state.cdse_auth=updated; token=updated["access_token"]
                continue
        if response.status_code in (429,502,503,504) and attempt<2:
            time.sleep(2**(attempt+1))
            continue
        break
    if response.status_code>=400:
        raise RuntimeError(f"CDSE HTTP {response.status_code}: {response_error(response)}")
    try:
        with MemoryFile(response.content) as mem:
            with mem.open() as ds:
                if ds.count!=bands:
                    raise ValueError("Unexpected band count")
                ds.read()
    except Exception as exc:
        raise RuntimeError("CDSE did not return a valid TIFF; response was not cached.") from exc
    tmp=path.with_suffix(".tif.part")
    try:
        tmp.write_bytes(response.content); tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)
    return path




def catalog_http(body):
    token=get_access_token()
    for attempt in range(2):
        try:
            response=requests.post(CATALOG_URL,json=body,
                headers={"Authorization":f"Bearer {token}"},timeout=(15,25))
        except requests.exceptions.RequestException as exc:
            if attempt<1:time.sleep(attempt+1);continue
            raise RuntimeError("Sentinel Hub catalogue connection failed: "+str(exc)) from exc
        if response.status_code==401 and attempt<1:
            auth=st.session_state.get("cdse_auth")
            refreshed=refresh_cdse_session(auth) if auth else None
            if refreshed:
                st.session_state.cdse_auth=refreshed;token=refreshed["access_token"];continue
        if response.status_code in (429,500,502,503,504) and attempt<1:
            time.sleep(attempt+1);continue
        if response.status_code>=400:
            raise RuntimeError(f"Sentinel Hub catalogue HTTP {response.status_code}: {response_error(response)}")
        return response.json()
    raise RuntimeError("Catalogue retry limit reached")


def type_for_product(product):
    if product.startswith("AI"):return "AER_AI"
    if product.startswith("CLOUD_"):return "CLOUD"
    return product


def dates_from_response(data,product_type=None):
    dates=set()
    for feature in data.get("features",[]):
        if isinstance(feature,str):
            stamp=feature
        elif isinstance(feature,dict):
            props=feature.get("properties",{})
            typ=props.get("s5p:type")
            if product_type and typ and typ!=product_type:continue
            stamp=props.get("datetime") or props.get("start_datetime") or feature.get("date") or props.get("date")
            if not stamp:raise ValueError("Catalogue returned records without acquisition dates.")
        else:
            raise ValueError("Unexpected catalogue distinct-date format")
        dates.add(date.fromisoformat(str(stamp)[:10]))
    return dates


def catalogue_dates(collection,aoi,start,end,product_type=None,force=False):
    key=fingerprint(["catalog-v10",collection,aoi,str(start),str(end),product_type])
    cache=CACHE_DIR/(key+".catalog.json")
    if cache.exists() and not force and time.time()-cache.stat().st_mtime<3600:
        saved=json.loads(cache.read_text());return [date.fromisoformat(x) for x in saved["days"]]
    body={"collections":[collection],"bbox":list(shape(aoi).bounds),
          "datetime":str(start)+"T00:00:00Z/"+str(end)+"T23:59:59Z",
          "limit":100,"distinct":"date"}
    if product_type:body["filter"]='"s5p:type" = '+repr(product_type)
    days=set();tokens=set()
    for page in range(200):
        data=catalog_http(body)
        days.update(dates_from_response(data,product_type))
        token=data.get("context",{}).get("next")
        if token is None:break
        serial=json.dumps(token,sort_keys=True)
        if serial in tokens:raise RuntimeError("Catalogue pagination repeated a token; availability is incomplete.")
        tokens.add(serial);body["next"]=token
    else:raise RuntimeError("Catalogue page limit reached; availability is incomplete.")
    output=sorted(d for d in days if start<=d<=end)
    atomic_json(cache,{"days":[str(x) for x in output],"source":CATALOG_URL})
    return output


def availability_for(config,force=False):
    start=date.fromisoformat(config["start"]);end=date.fromisoformat(config["end"])
    keys=(["S2"] if config["smoke"] else [])+config["products"]
    output={};dedup={};shared_failure=None
    for key in keys:
        if shared_failure:
            output[key]={"ok":False,"days":[],"error":"Catalogue checks stopped after shared service/authentication failure: "+shared_failure}
            continue
        collection="sentinel-2-l2a" if key=="S2" else "sentinel-5p-l2"
        product_type=None if key=="S2" else type_for_product(key)
        identity=(collection,product_type)
        if identity not in dedup:
            try:
                dates=catalogue_dates(collection,config["aoi"],start,end,product_type,force)
                dedup[identity]={"ok":True,"days":[str(d) for d in dates],"error":None}
            except Exception as exc:
                message=str(exc)
                dedup[identity]={"ok":False,"days":[],"error":message}
                if any(x in message for x in ("HTTP 401","HTTP 403","HTTP 429","HTTP 502","HTTP 503","HTTP 504","connection failed")):
                    shared_failure=message
        output[key]=dict(dedup[identity])
    return output


def preferred_catalogue_date(available,fallback):
    groups=[set(v["days"]) for v in available.values() if v.get("ok") and v.get("days")]
    if not groups:return str(fallback)
    common=set.intersection(*groups)
    return max(common if common else set.union(*groups))


def latest_available_before(config):
    end=date.fromisoformat(config["end"])
    start=end-timedelta(days=180)
    errors=[]
    # Probe selected priority products only; no dates or measurements are fabricated.
    probes=[p for p in ("CH4","CO") if p in config["products"]]+(["S2"] if config["smoke"] else [])
    for product in config["products"]:
        if product not in probes:probes.append(product)
    found=[]
    for key in probes:
        try:
            ds=catalogue_dates("sentinel-2-l2a" if key=="S2" else "sentinel-5p-l2",config["aoi"],start,end,
                              None if key=="S2" else type_for_product(key))
            if ds:found.append(max(ds))
        except Exception as exc:errors.append(f"{key}: {exc}")
        if found:break
    return (max(found) if found else None),errors


def row_pixel_area_km2(profile,h,w):
    if profile["crs"].is_geographic:
        t=profile["transform"];geod=Geod(ellps="WGS84")
        result=np.empty((h,1),dtype=np.float64)
        for row in range(h):
            west=t.c;east=west+t.a;top=t.f+row*t.e;bottom=top+t.e
            area,_=geod.polygon_area_perimeter([west,east,east,west],[top,top,bottom,bottom])
            result[row,0]=abs(area)/1e6
        return result
    t=profile["transform"]
    return np.full((h,1),abs(t.a*t.e-t.b*t.d)/1e6)


def finite_text(value):
    if value is None:return "No valid observation"
    try:return f"{float(value):.7g}" if np.isfinite(value) else "No valid observation"
    except (TypeError,ValueError):return "No valid observation"


def readable_table(frame):
    table=frame.copy()
    for col in table.columns:
        if pd.api.types.is_numeric_dtype(table[col]):
            table[col]=table[col].map(lambda v:finite_text(v) if pd.notna(v) else "—")
    return table


def s2_script(day):
    epoch=(day-EPOCH).days
    return """//VERSION=3
function setup(){return {input:[{bands:["B02","B03","B04","B08","B11","B12","SCL","dataMask"],
 units:["REFLECTANCE","REFLECTANCE","REFLECTANCE","REFLECTANCE","REFLECTANCE","REFLECTANCE","DN","DN"]}],
 mosaicking:"SIMPLE",output:{bands:9,sampleType:"FLOAT32",nodataValue:-9999}};}
function evaluatePixel(s){
 if(!(Number(s.dataMask)>0))return [-9999,-9999,-9999,-9999,-9999,-9999,0,0,-9999];
 return [s.B02,s.B03,s.B04,s.B08,s.B11,s.B12,s.SCL,1,__DAY__];}
""".replace("__DAY__",str(epoch))




def request_s2_day(day,config,force=False):
    g=config["grid"]
    payload={"input":{"bounds":{"bbox":g["bounds"],"properties":{"crs":"http://www.opengis.net/def/crs/EPSG/0/4326"}},
        "data":[{"type":"sentinel-2-l2a","dataFilter":{"timeRange":{
            "from":str(day)+"T00:00:00Z","to":str(day)+"T23:59:59Z"},
            "maxCloudCoverage":100,"mosaickingOrder":"leastCC"},"processing":{"upsampling":"NEAREST","downsampling":"NEAREST"}}]},
        "output":{"width":g["w"],"height":g["h"],"responses":[{"identifier":"default","format":{"type":"image/tiff"}}]},
        "evalscript":s2_script(day)}
    return process_request(payload,fingerprint(["s2-daily-v10",payload]),9,force=force)


def merge_daily(existing,incoming,day,medium=False):
    reject=[0,1,3,9,10,11] if medium else [0,1,3,8,9,10,11]
    def score(array):
        observed=(array[7]>.5)&np.isfinite(array[:6]).all(axis=0)&(array[:6]!=-9999).all(axis=0)
        age=(day-EPOCH).days-array[8]
        observed &= np.isfinite(age)&(age>=0)&(age<30)
        good=~np.isin(array[6],reject)
        return np.where(observed,np.where(good,0,1000)+age,np.inf)
    if existing is None:
        result=incoming.copy()
        invalid=~np.isfinite(score(incoming))
        result[:,invalid]=-9999
        result[6:8,invalid]=0
        return result
    if incoming.shape!=existing.shape:raise ValueError("S2 daily grids do not match")
    use=score(incoming)<score(existing)
    existing[:,use]=incoming[:,use]
    return existing


def request_s2(day,config,force=False,status=None):
    key=fingerprint(["s2-catalogued-mosaic-v10",day,config["aoi"],config["grid"],config["params"]["medium_cloud"]])
    path=CACHE_DIR/(key+".tif")
    if path.exists() and not force:return path
    notes=[]
    try:
        dates=catalogue_dates("sentinel-2-l2a",config["aoi"],day-timedelta(days=29),day)
        selected=sorted(dates,reverse=True)[:7]
    except Exception as exc:
        # Unlike v9, a catalogue failure does not stop all older-day attempts.
        notes.append("Catalogue failed. Bounded daily fallback used: "+str(exc))
        selected=[day-timedelta(days=n) for n in range(7)]
    if not selected:
        raise RuntimeError("Catalogue search completed but found no S2 acquisition in the preceding 30 days for this AOI.")
    merged=None;profile=None;used=[];failures=[];consecutive_failures=0
    for candidate in selected:
        if status is not None:status.caption(f"S2 · source acquisition {candidate}")
        try:
            daily=request_s2_day(candidate,config,force=force)
            with rasterio.open(daily) as ds:
                array=ds.read().astype(np.float32)
                if profile is not None and (ds.crs!=profile["crs"] or ds.transform!=profile["transform"]):
                    raise ValueError("S2 CRS/transform mismatch")
                profile=ds.profile.copy()
            merged=merge_daily(merged,array,day,config["params"]["medium_cloud"])
            used.append(str(candidate));consecutive_failures=0
            inside=clip_mask(profile,*merged.shape[1:],config["aoi"])
            reject=[0,1,3,9,10,11] if config["params"]["medium_cloud"] else [0,1,3,8,9,10,11]
            good=(merged[7]>.5)&inside&~np.isin(merged[6],reject)
            if good.sum()/max(1,inside.sum())>=0.98:break
        except Exception as exc:
            failures.append(f"{candidate}: {exc}")
            consecutive_failures+=1
            if consecutive_failures>=2:
                notes.append("Stopped after two consecutive source request failures; retry is available.")
                break
    if merged is None:raise RuntimeError("Every S2 source request failed: "+" | ".join(failures))
    prof=profile.copy();prof.update(count=9,dtype="float32",nodata=-9999,compress="deflate",tiled=False)
    tmp=path.with_suffix(".tif.part")
    try:
        with rasterio.open(tmp,"w",**prof) as ds:ds.write(merged)
        tmp.replace(path)
    finally:tmp.unlink(missing_ok=True)
    atomic_json(path.with_suffix(".context.json"),{"used_dates":used,"notes":notes,"source_failures":failures,
        "catalogue":CATALOG_URL,"method":"daily requests + acquisition-date preserving mosaic","max_acquisition_days":7})
    return path


def read_s2(path,aoi):
    with rasterio.open(path) as ds:
        array=ds.read().astype(np.float32);profile=ds.profile.copy()
    inside=clip_mask(profile,*array.shape[1:],aoi)
    array[:6][array[:6]==-9999]=np.nan
    observed=(array[7]>.5)&inside&np.isfinite(array[:6]).all(axis=0)
    bands={key:array[i] for i,key in enumerate(BANDS)}
    bands["dataMask"]=observed.astype(np.float32)
    for key in BANDS:
        if key!="dataMask":bands[key][~observed]=np.nan
    return bands,profile,inside


def nd(a,b):
    total=a+b
    return np.divide(a-b,total,out=np.full_like(a,np.nan),where=np.isfinite(total)&(np.abs(total)>1e-6))


def smoke_screen(bands,day,params):
    b,g,r,n,sw1,sw2=[bands[k] for k in BANDS[:6]]
    observed=bands["dataMask"]>0.5
    scl=np.nan_to_num(bands["SCL"],nan=0).astype(np.int32)
    invalid=[0,1,3,9,10,11] if params["medium_cloud"] else [0,1,3,8,9,10,11]
    quality=observed & ~np.isin(scl,invalid)
    finite=np.logical_and.reduce([np.isfinite(bands[k]) for k in BANDS[:6]])
    quality &= finite
    index=nd(g,sw1); bright=(b+g+r)/3
    ratio=np.divide(sw2,sw1,out=np.full_like(sw1,np.nan),where=sw1>1e-6)
    vegetation=nd(n,r)
    ndwi=nd(g,n)
    water=(scl==6)|((ndwi>0.0)&(n<0.20))
    candidate=quality & (~water if params.get("exclude_water",True) else True) & (index>params["threshold"]) & (bright>params["brightness"]) & (ratio<params["swir_ratio"]) & (vegetation<0.45)
    if params["smooth"]>0:
        weights=gaussian_filter(quality.astype(np.float32),sigma=params["smooth"])
        values=gaussian_filter(np.where(quality,index,0),sigma=params["smooth"])
        smooth=np.divide(values,weights,out=np.full_like(values,np.nan),where=weights>0.05)
        candidate &= smooth>params["threshold"]
    components,count=label(candidate,structure=np.ones((3,3)))
    sizes=np.bincount(components.ravel(),minlength=count+1)
    keep=np.flatnonzero(sizes>=params["min_pixels"]);keep=keep[keep>0]
    smoke=np.isin(components,keep)&quality
    if params.get("exclude_water",True):smoke&=~water
    if params["dilation"] and smoke.any():
        smoke=binary_dilation(smoke,structure=disk(params["dilation"]))&quality
        if params.get("exclude_water",True):smoke&=~water
    age=(day-EPOCH).days-bands["observation_day"]
    fresh=quality & (age>=0) & (age<1)
    return {"observed":observed,"quality":quality,"smoke":smoke,"fresh":fresh,
            "current":smoke&fresh,"historical":smoke&~fresh,"age":np.where(observed,age,np.nan),
            "index":index}


def s5_script(band):
    return """//VERSION=3
function setup(){return {input:[{bands:["__BAND__","dataMask"]}],
 mosaicking:"SIMPLE",output:{bands:3,sampleType:"FLOAT32",nodataValue:-9999}};}
function evaluatePixel(s){var v=s.__BAND__;var ok=Number(s.dataMask)>0 && isFinite(v) && v!==-9999;
 return ok?[v,1,1]:[-9999,0,0];}
""".replace("__BAND__",band)


def request_s5(day,product,config,force=False,qa_override=None):
    bounds=shape(config["aoi"]).bounds
    ratio=max(0.1,(bounds[2]-bounds[0])*math.cos(math.radians((bounds[1]+bounds[3])/2))/max(1e-6,bounds[3]-bounds[1]))
    w=max(1,round(420*min(1,ratio)));h=max(1,round(420*min(1,1/ratio)))
    qa=config["qa"] if config["qa_override"] else (75 if product=="NO2" else 50)
    if qa_override is not None:qa=qa_override
    payload={"input":{"bounds":{"bbox":list(bounds),"properties":{"crs":"http://www.opengis.net/def/crs/EPSG/0/4326"}},
        "data":[{"type":"sentinel-5p-l2","dataFilter":{"timeRange":{
            "from":str(day)+"T00:00:00Z","to":str(day)+"T23:59:59Z"},"mosaickingOrder":"mostRecent"},
            "processing":{"minQa":qa,"upsampling":"NEAREST"}}]},
        "output":{"width":w,"height":h,"responses":[{"identifier":"default","format":{"type":"image/tiff"}}]},
        "evalscript":s5_script(ALL_PRODUCTS[product]["band"])}
    return process_request(payload,fingerprint(["s5-catalogued-direct-v10",payload]),3,force=force)


def read_s5(path,aoi):
    with rasterio.open(path) as ds:
        values=ds.read(1).astype(np.float32);mask=ds.read(2)>.5;profile=ds.profile.copy()
    inside=clip_mask(profile,*values.shape,aoi)
    valid=mask&inside&np.isfinite(values)&(values!=-9999)
    values[~valid]=np.nan
    return values,profile,inside


def tif_bytes(array,profile):
    p=profile.copy();p.update(count=1,dtype="float32",nodata=-9999,compress="deflate",tiled=False)
    with MemoryFile() as memory:
        with memory.open(**p) as ds:
            ds.write(np.where(np.isfinite(array),array,-9999).astype(np.float32),1)
        return memory.read()


def png_bytes(array):
    buffer=io.BytesIO();Image.fromarray(array).save(buffer,format="PNG")
    return buffer.getvalue()


def rgb_rgba(bands,inside):
    rgb=np.stack([bands["B04"],bands["B03"],bands["B02"]],axis=-1)
    rgb=(np.power(np.clip(np.nan_to_num(rgb)/0.30,0,1),0.8)*255).astype(np.uint8)
    alpha=np.where((bands["dataMask"]>0.5)&inside,255,0).astype(np.uint8)
    return np.concatenate([rgb,alpha[:,:,None]],axis=2)


def field_rgba(values,limits,cmap,opacity=210):
    low,high=limits
    if high<=low:high=low+max(1e-9,abs(low)*0.01)
    norm=np.clip((np.nan_to_num(values,nan=low)-low)/(high-low),0,1)
    rgba=(plt.get_cmap(cmap)(norm)*255).astype(np.uint8)
    rgba[:,:,3]=np.where(np.isfinite(values),opacity,0)
    return rgba


def mask_rgba(mask,color):
    rgba=np.zeros((*mask.shape,4),dtype=np.uint8)
    rgba[mask]=[*color,210]
    return rgba


def colorbar_png(low,high,cmap,unit):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.colors import Normalize
    from matplotlib.colorbar import ColorbarBase
    fig=Figure(figsize=(6.5,0.8),dpi=125,facecolor="#f1faee",layout="none")
    canvas=FigureCanvasAgg(fig);ax=fig.add_axes([0.05,0.54,0.9,0.25])
    ColorbarBase(ax,cmap=plt.get_cmap(cmap),norm=Normalize(low,high),orientation="horizontal")
    ax.tick_params(labelsize=8);ax.set_xlabel(unit,fontsize=9)
    buf=io.BytesIO();canvas.print_png(buf)
    return buf.getvalue()


def mercator_image(rgba,profile):
    h,w=rgba.shape[:2]
    native_bounds=rasterio.transform.array_bounds(h,w,profile["transform"])
    tx,dw,dh=calculate_default_transform(profile["crs"],"EPSG:3857",w,h,*native_bounds)
    dst=np.zeros((4,dh,dw),dtype=np.uint8)
    for i in range(4):
        reproject(rgba[:,:,i],dst[i],src_transform=profile["transform"],src_crs=profile["crs"],
          dst_transform=tx,dst_crs="EPSG:3857",resampling=Resampling.nearest)
    bounds=rasterio.transform.array_bounds(dh,dw,tx)
    west,south,east,north=transform_bounds("EPSG:3857","EPSG:4326",*bounds)
    url="data:image/png;base64,"+base64.b64encode(png_bytes(np.moveaxis(dst,0,-1))).decode()
    return url,[[south,west],[north,east]]


def add_raster(fmap,rgba,profile,name,show=True,opacity=1):
    url,bounds=mercator_image(rgba,profile)
    folium.raster_layers.ImageOverlay(url,bounds=bounds,name=name,show=show,opacity=opacity).add_to(fmap)


def new_result_map(aoi):
    g=shape(aoi);c=g.centroid
    fmap=folium.Map([c.y,c.x],tiles=None,control_scale=True)
    folium.TileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png",
        name="OpenStreetMap — geographic context",attr="© OpenStreetMap contributors",max_zoom=19).add_to(fmap)
    return fmap


def finish_map(fmap,aoi,key,height=580):
    g=shape(aoi)
    folium.GeoJson(aoi,name="Selected AOI",style_function=lambda _: {"color":"#457b9d","weight":2,"fill":False}).add_to(fmap)
    folium.LayerControl(collapsed=False).add_to(fmap)
    fmap.fit_bounds([[g.bounds[1],g.bounds[0]],[g.bounds[3],g.bounds[2]]],padding=(10,10))
    MousePosition(position="bottomright",prefix="Lat/Lon",num_digits=5).add_to(fmap)
    st_folium(fmap,key=key,height=height,use_container_width=True,returned_objects=[])


def package_png(rgba,profile):
    t=profile["transform"]
    world=f"{t.a:.12f}\n{t.d:.12f}\n{t.b:.12f}\n{t.e:.12f}\n{t.c+(t.a+t.b)/2:.12f}\n{t.f+(t.d+t.e)/2:.12f}\n"
    buf=io.BytesIO()
    with zipfile.ZipFile(buf,"w",zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("map.png",png_bytes(rgba));archive.writestr("map.pgw",world)
        archive.writestr("map.prj",profile["crs"].to_wkt())
    return buf.getvalue()


def session_folder(config):
    folder=RESULT_DIR/fingerprint(config);folder.mkdir(parents=True,exist_ok=True)
    return folder


def load_record(day,config):
    path=session_folder(config)/(str(day)+".json")
    if path.exists():
        try:return json.loads(path.read_text(encoding="utf-8"))
        except Exception:pass
    return {"date":str(day),"products":{},"errors":{}}


def process_day(day,config,status=None,retry=False):
    record=load_record(day,config)
    if retry:record={"date":str(day),"products":{},"errors":{}}
    record.setdefault("tracebacks",{})
    record["attempted_utc"]=datetime.utcnow().isoformat()+"Z"
    available=config.get("availability",{})
    if config["smoke"] and ("s2" not in record or record["errors"].get("S2")):
        if status is not None:status.caption(f"{day} · S2 actual acquisition dates")
        try:
            path=request_s2(day,config,force=retry,status=status)
            bands,profile,inside=read_s2(path,config["aoi"])
            if not inside.any():raise ValueError("Output grid does not intersect AOI")
            result=smoke_screen(bands,day,config["params"])
            pixel_area=row_pixel_area_km2(profile,*result["current"].shape)
            denom=max(1,int(inside.sum()))
            context_path=path.with_suffix(".context.json")
            record["s2"]={"path":str(path),"coverage":float(result["observed"].sum()/denom*100),
                "quality_coverage":float(result["quality"].sum()/denom*100),
                "current_coverage":float(result["fresh"].sum()/denom*100),
                "current_area":float((result["current"]*pixel_area).sum()) if result["fresh"].any() else None,
                "historical_area":float((result["historical"]*pixel_area).sum()) if result["quality"].any() else None,
                "current_pixels":int(result["current"].sum()),"observed_pixels":int(result["observed"].sum()),
                "current_assessable_pixels":int(result["fresh"].sum()),
                "candidate_fraction":float(result["smoke"].sum()/max(1,result["quality"].sum())),
                "context":json.loads(context_path.read_text()) if context_path.exists() else {},
                "status":"same-day observed" if result["fresh"].any() else "older context only" if result["observed"].any() else "no valid returned image"}
            record["errors"].pop("S2",None);record["tracebacks"].pop("S2",None)
        except Exception as exc:
            record.pop("s2",None);record["errors"]["S2"]=str(exc);record["tracebacks"]["S2"]=traceback.format_exc()
        atomic_json(session_folder(config)/(str(day)+".json"),record)
    for product in config["products"]:
        if product in record["products"] and not record["errors"].get(product):continue
        availability=available.get(product,{})
        if availability.get("ok") and str(day) not in availability.get("days",[]):
            record["products"][product]={"path":None,"coverage":None,"mean":None,"min":None,"max":None,
                "valid_cells":0,"status":"no catalogue source on this day","available_days":availability.get("days",[])}
            record["errors"].pop(product,None)
            atomic_json(session_folder(config)/(str(day)+".json"),record)
            continue
        if status is not None:status.caption(f"{day} · {ALL_PRODUCTS[product]['label']} · direct product")
        try:
            path=request_s5(day,product,config,force=retry)
            values,profile,inside=read_s5(path,config["aoi"])
            if not inside.any():raise ValueError("S5P grid does not intersect selected AOI")
            finite=values[np.isfinite(values)]
            item={"path":str(path),"coverage":float(finite.size/max(1,inside.sum())*100),
                "mean":float(finite.mean()) if finite.size else None,
                "min":float(finite.min()) if finite.size else None,"max":float(finite.max()) if finite.size else None,
                "valid_cells":int(finite.size),"all_zero":bool(finite.size and np.all(finite==0)),
                "status":"valid direct retrieval" if finite.size else "catalogue source found, but no QA-valid pixels returned",
                "available_days":availability.get("days",[])}
            standard_qa=config["qa"] if config["qa_override"] else (75 if product=="NO2" else 50)
            if not finite.size and standard_qa>0:
                try:
                    probe=request_s5(day,product,config,force=retry,qa_override=0)
                    low,_,low_inside=read_s5(probe,config["aoi"])
                    low_values=low[np.isfinite(low)]
                    item["qa0_probe"]={"path":str(probe),"valid_cells":int(low_values.size),
                        "mean":float(low_values.mean()) if low_values.size else None,
                        "coverage":float(low_values.size/max(1,low_inside.sum())*100),
                        "note":"Diagnostic QA=0 only; excluded from the quality-filtered monthly series."}
                except Exception as exc:item["qa0_probe_error"]=str(exc)
            record["products"][product]=item
            record["errors"].pop(product,None);record["tracebacks"].pop(product,None)
        except Exception as exc:
            record["products"].pop(product,None);record["errors"][product]=str(exc);record["tracebacks"][product]=traceback.format_exc()
        atomic_json(session_folder(config)/(str(day)+".json"),record)
    return record


def table_for(config):
    rows=[]
    for stamp in pd.date_range(config["start"],config["end"]):
        record=load_record(stamp.date(),config);s2=record.get("s2",{})
        row={"date":stamp,"S2_status":record["errors"].get("S2",s2.get("status","not requested")),
             "S2_current_candidate_km2":s2.get("current_area",np.nan),
             "S2_current_coverage_pct":s2.get("current_coverage",np.nan),"S2_context_coverage_pct":s2.get("coverage",np.nan)}
        for product in config["products"]:
            p=record["products"].get(product,{})
            row[product]=p.get("mean",np.nan);row[product+"_coverage_pct"]=p.get("coverage",np.nan)
            row[product+"_status"]=record["errors"].get(product,p.get("status","not requested"))
        rows.append(row)
    table=pd.DataFrame(rows).set_index("date")
    for column in table:
        if not column.endswith("_status"):table[column]=pd.to_numeric(table[column],errors="coerce")
    return table


def fixed_limits(product,config):
    # If a full month is prepared use its pooled distribution; otherwise use
    # a frozen physical display range. Neither changes when the slider moves.
    limits_file=session_folder(config)/"scales.json"
    if limits_file.exists():
        saved=json.loads(limits_file.read_text())
        if product in saved:return saved[product]
    return ALL_PRODUCTS[product]["range"]


def lock_month_scales(config):
    saved={}
    for product in config["products"]:
        chunks=[]
        for stamp in pd.date_range(config["start"],config["end"]):
            layer=load_record(stamp.date(),config)["products"].get(product)
            if layer and layer.get("path"):
                values,_,_=read_s5(layer["path"],config["aoi"])
                finite=values[np.isfinite(values)]
                if finite.size:chunks.append(finite[::max(1,finite.size//1500)])
        if chunks:
            lo,hi=np.percentile(np.concatenate(chunks),[2,98])
            if hi<=lo:
                span=max(abs(float(lo))*.02,1e-9);lo-=span;hi+=span
            saved[product]=[float(lo),float(hi)]
    atomic_json(session_folder(config)/"scales.json",saved)


def find_catalog(aoi,start,end):
    dates=catalogue_dates("sentinel-2-l2a",aoi,start,end,force=True)
    return [{"id":"S2_"+str(d),"properties":{"datetime":str(d)+"T12:00:00Z"}} for d in dates]


st.set_page_config(page_title="Smoke & Methane Monitor",page_icon="🛰️",layout="wide",initial_sidebar_state="collapsed")
st.markdown('<style>\n:root {\n    --red: #e63946;\n    --honeydew: #f1faee;\n    --frost: #a8dadc;\n    --blue: #457b9d;\n    --navy: #1d3557;\n    --black: #111111;\n    --white: #ffffff;\n    --border: #d8e6e8;\n    --muted: #4f5d63;\n    --dark-field: #292a33;\n}\n.stApp { background: #f1faee; color: #111111 !important; }\n[data-testid="stHeader"] { background: #f1faee !important; height: 3.25rem !important; }\n[data-testid="stSidebar"] { display: none; }\n.block-container { max-width: 1700px; padding-top: 3.9rem !important; padding-bottom: 0.8rem; padding-left: 1.2rem; padding-right: 1.2rem; }\n.app-header { position: relative; z-index: 10; display: flex; align-items: center; justify-content: space-between; background: #ffffff; border: 1px solid var(--border); border-radius: 16px; padding: 0.75rem 1rem; margin-top: 0.15rem; margin-bottom: 0.9rem; box-shadow: 0 2px 10px rgba(29,53,87,0.05); }\n.app-title { color: #111111 !important; font-size: 1.45rem; font-weight: 850; line-height: 1.1; }\n.app-subtitle { color: #111111 !important; font-size: 0.78rem; margin-top: 0.15rem; }\n.status-pill { background: #f1faee; color: #111111 !important; border: 1px solid #a8dadc; border-radius: 999px; padding: 0.35rem 0.7rem; font-size: 0.72rem; font-weight: 750; white-space: nowrap; }\n.app-card { background: #ffffff; border: 1px solid var(--border); border-radius: 15px; padding: 0.75rem; box-shadow: 0 2px 10px rgba(29,53,87,0.04); height: 100%; color: #111111 !important; }\n.card-title { color: #111111 !important; font-size: 1rem; font-weight: 800; margin-bottom: 0.1rem; }\n.card-caption { color: #111111 !important; font-size: 0.73rem; margin-bottom: 0.45rem; }\n.section-label { display: inline-block; background: #a8dadc; color: #111111 !important; border-radius: 999px; padding: 0.2rem 0.55rem; font-size: 0.65rem; font-weight: 800; letter-spacing: 0.03em; margin-bottom: 0.35rem; }\n.stApp p, .stApp label, .stApp small, .stApp strong, .stApp em, .stApp li, .stApp td, .stApp th,\n.stApp [data-testid="stMarkdownContainer"], .stApp [data-testid="stMarkdownContainer"] p,\n.stApp [data-testid="stMarkdownContainer"] span, .stApp [data-testid="stMarkdownContainer"] li { color: #111111 !important; }\ndiv[data-testid="stDateInput"] input, .stDateInput input,\ndiv[data-testid="stNumberInput"] input, .stNumberInput input {\n    background-color: var(--dark-field) !important; color: #ffffff !important;\n    -webkit-text-fill-color: #ffffff !important; caret-color: #ffffff !important;\n}\ninput::placeholder, textarea::placeholder { color: #bfc3cc !important; opacity: 1 !important; }\ndiv[data-baseweb="select"] input, div[data-baseweb="select"] [role="combobox"], div[data-baseweb="select"] * { color: #111111 !important; }\ndiv[data-baseweb="popover"] [role="listbox"], div[data-baseweb="popover"] ul[role="listbox"],\ndiv[data-baseweb="popover"] [role="option"], div[data-baseweb="popover"] li[role="option"] { background: #111318 !important; }\ndiv[data-baseweb="popover"] [role="option"] *, ul[role="listbox"] *, li[role="option"] * { color: #ffffff !important; -webkit-text-fill-color: #ffffff !important; }\ndiv[data-baseweb="popover"] [role="option"]:hover, div[data-baseweb="popover"] li[role="option"]:hover { background: #2b2e38 !important; }\n.stDateInput, .stSlider, .stNumberInput, .stSelectbox { margin-bottom: 0.15rem; }\n.stSlider label, .stSlider [data-testid="stTickBar"] * { color: #111111 !important; }\n.stSlider [data-testid="stThumbValue"], .stSlider [data-testid="stThumbValue"] * { color: #ffffff !important; }\n[data-baseweb="calendar"] *, [data-baseweb="popover"] [data-baseweb="calendar"] *, [data-baseweb="calendar"] button { color: #ffffff !important; }\ninput:-webkit-autofill, input:-webkit-autofill:hover, input:-webkit-autofill:focus { -webkit-text-fill-color: #ffffff !important; caret-color: #ffffff !important; }\n.auth-card { background: #f8fbfb; border: 1px solid #d7e4e7; border-radius: 11px; padding: 0.65rem 0.75rem; margin-top: 0.45rem; }\n.auth-status { background: #e8f7ea; border: 1px solid #9ed2a4; color: #155724 !important; border-radius: 9px; padding: 0.45rem 0.6rem; font-size: 0.76rem; font-weight: 700; margin-bottom: 0.45rem; }\n.auth-help { color: #111111 !important; font-size: 0.72rem; line-height: 1.45; margin: 0.2rem 0 0.45rem 0; }\n.stButton > button, .stDownloadButton > button { border-radius: 9px; min-height: 2.15rem; font-weight: 750; font-size: 0.78rem; color: #111111 !important; }\n.stButton > button[kind="primary"] { background: #e63946; border-color: #e63946; color: #ffffff !important; }\n.stButton > button[kind="primary"] *, .stDownloadButton > button[kind="primary"] * { color: #ffffff !important; }\n.stButton > button[kind="primary"]:hover { background: #c92f3b; border-color: #c92f3b; color: #ffffff !important; }\n.stDownloadButton > button { background: #ffffff; color: #111111 !important; border: 1px solid #a8dadc; }\n.stDownloadButton > button:hover { background: #f1faee; border-color: #457b9d; color: #111111 !important; }\ndiv[data-testid="stDataFrame"] { border: 1px solid var(--border); }\ndiv[data-testid="stDataFrame"] * { color: #111111 !important; }\n.result-card { background: #ffffff; border: 1px solid #d7e4e7; border-radius: 12px; padding: 0.7rem; height: 100%; }\n.result-tag { display: inline-block; background: #a8dadc; color: #111111 !important; border-radius: 999px; padding: 0.15rem 0.5rem; font-size: 0.62rem; font-weight: 800; margin-bottom: 0.25rem; }\n.result-name { color: #111111 !important; font-size: 0.85rem; font-weight: 800; margin-bottom: 0.35rem; }\n.result-note { background: #ffffff; border: 1px solid #d7e4e7; border-radius: 10px; padding: 0.7rem 0.8rem; color: #111111 !important; font-size: 0.76rem; line-height: 1.55; margin-top: 0.5rem; }\n.download-label { color: #111111 !important; font-size: 0.62rem; font-weight: 700; margin: 0.2rem 0 0.12rem 0; }\n.map-frame { border: 1px solid var(--border); border-radius: 10px; overflow: hidden; }\n.result-legend { background: #ffffff; border: 1px solid #d7e4e7; border-radius: 10px; padding: 0.75rem 0.7rem; min-height: 96px; box-sizing: border-box; display: flex; flex-direction: column; justify-content: center; gap: 0.42rem; }\n.result-legend .legend-heading { color: #111111 !important; font-size: 0.88rem; font-weight: 800; }\n.result-legend .legend-row { display: flex; align-items: center; gap: 0.45rem; color: #111111 !important; font-size: 0.82rem; line-height: 1.25; }\n.result-legend .legend-row span:last-child { color: #111111 !important; }\n.legend-swatch { width: 18px; height: 14px; min-width: 18px; border: 1px solid #555; border-radius: 2px; display: inline-block; }\n.coords-panel { margin-top: 0.55rem; background: #f8fbfb; border: 1px solid #d7e4e7; border-radius: 10px; padding: 0.5rem 0.7rem; display: flex; flex-direction: column; gap: 0.3rem; }\n.coord-row { display: flex; justify-content: space-between; align-items: center; gap: 0.5rem; font-size: 0.78rem; color: #111111 !important; }\n.coord-label { font-weight: 700; color: #111111 !important; flex: 1 1 auto; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }\n.coord-value { flex: 0 0 auto; font-family: ui-monospace, "SFMono-Regular", Menlo, Consolas, monospace; background: #eef5f6; border: 1px solid #cfe0e3; border-radius: 6px; padding: 0.12rem 0.5rem; color: #111111 !important; font-size: 0.74rem; white-space: nowrap; }\nfooter { visibility: hidden; }\n.stMarkdown { margin-bottom: 0.1rem; }\n.element-container { margin-bottom: 0.15rem; }\n</style>',unsafe_allow_html=True)
st.markdown("""<style>
.result-shell{background:white;border:1px solid #d8e6e8;border-radius:18px;padding:18px;margin:12px 0;}
.result-head{font-size:1.25rem;font-weight:850;color:#1d3557;margin-bottom:5px;}
.result-sub{font-size:.8rem;color:#4f5d63;margin-bottom:12px;}
.result-chip{display:inline-block;padding:5px 10px;border-radius:30px;background:#e7f3f3;margin:3px;font-size:.75rem;}
[data-testid="stMetric"]{background:#fff;border:1px solid #d8e6e8;border-radius:12px;padding:12px;}
[data-testid="stMetricLabel"]{font-size:.8rem;}
[data-testid="stMetricValue"]{font-size:1.65rem;}
[data-testid="stTabs"] [data-baseweb="tab"]{font-weight:750;font-size:.85rem;}
</style>""",unsafe_allow_html=True)
st.markdown("""<style>
:root{color-scheme:light;}
.stApp button[data-testid="stBaseButton-secondary"],.stApp button[kind="secondary"],
.stApp [data-testid="stBaseButton-secondaryFormSubmit"],.stApp .stDownloadButton button,
.stApp [data-testid="stDownloadButton"] button,.stApp a[data-testid="stBaseLinkButton-secondary"],
.stApp .stButton>button:not([kind="primary"]){
 background:#ffffff !important;background-color:#ffffff !important;border:1.5px solid #457b9d !important;
 color:#111111 !important;-webkit-text-fill-color:#111111 !important;opacity:1 !important;box-shadow:0 1px 3px rgba(29,53,87,.12);}
.stApp button[data-testid="stBaseButton-secondary"] *,.stApp button[kind="secondary"] *,
.stApp .stDownloadButton button *,.stApp [data-testid="stDownloadButton"] button *,
.stApp a[data-testid="stBaseLinkButton-secondary"] *,.stApp .stButton>button:not([kind="primary"]) *{
 color:#111111 !important;-webkit-text-fill-color:#111111 !important;}
.stApp button[data-testid="stBaseButton-secondary"]:hover,.stApp .stDownloadButton button:hover,
.stApp a[data-testid="stBaseLinkButton-secondary"]:hover,.stApp .stButton>button:not([kind="primary"]):hover{
 background:#a8dadc !important;border-color:#1d3557 !important;}
.stApp button:disabled,.stApp button[disabled]{opacity:.55 !important;cursor:not-allowed;}
.stApp .stButton>button:not([kind="primary"]):disabled{background:#e8eef0 !important;}
.stApp button[kind="primary"],.stApp button[data-testid="stBaseButton-primary"]{
 background:#e63946 !important;border:1.5px solid #e63946 !important;color:#ffffff !important;}
</style>""",unsafe_allow_html=True)
st.markdown("""<div class="app-header"><div><div class="app-title">🛰️ Smoke & Methane Monitor</div>
<div class="app-subtitle">Sentinel-2 smoke context · Sentinel-5P methane and atmospheric products · 30-day explorer</div></div>
<div class="status-pill">One AOI · One timeline · No reference selector</div></div>""",unsafe_allow_html=True)
if "aoi" not in st.session_state:st.session_state.aoi=mapping(DEFAULT_AOI)
map_col,control_col=st.columns([1.65,1],gap="small")
with map_col:
    st.markdown('<div class="section-label">01 · STUDY AREA</div>',unsafe_allow_html=True)
    st.markdown('<div class="card-title">Area of Interest</div>',unsafe_allow_html=True)
    qcol,bcol=st.columns([4,1])
    with qcol:query=st.text_input("Location search",placeholder="Strait of Hormuz, Bandar Abbas, Bushehr, …",label_visibility="collapsed")
    with bcol:find=st.button("🔍 Find",use_container_width=True)
    if find:
        if not query.strip():st.warning("Type a location name first.")
        else:
            with st.spinner("Searching location…"):geo=geocode_location(query.strip())
            if geo:
                st.session_state.search_center=[geo[0],geo[1]]
                st.session_state.search_name=geo[2]
            else:st.warning("Location not found.")
    if st.session_state.get("search_center"):
        ic,cc=st.columns([5,1])
        ic.caption(st.session_state.get("search_name","Searched location"))
        if cc.button("✖ Clear"):
            st.session_state.pop("search_center",None);st.session_state.pop("search_name",None);st.rerun()
    mapdata=st_folium(create_map(st.session_state.aoi,st.session_state.get("search_center"),9),
                       height=430,use_container_width=True,key="aoi_map")
    if mapdata and mapdata.get("all_drawings"):
        geom=normalize_geometry({"type":"FeatureCollection","features":mapdata["all_drawings"]})
        if geom:st.session_state.aoi=geom
    g=shape(ensure_aoi(st.session_state.aoi));c=g.centroid
    st.caption(f"🎯 AOI centre: Lat {c.y:.5f} · Lon {c.x:.5f} | Draw a rectangle/polygon to replace the study area.")
    if mapdata and mapdata.get("last_clicked"):
        last=mapdata["last_clicked"];st.caption(f"🖱 Last click: {last['lat']:.5f}, {last['lng']:.5f}")
    if st.session_state.get("search_center"):
        sc=st.session_state.search_center;st.caption(f"📍 Search location: {sc[0]:.5f}, {sc[1]:.5f}")
with control_col:
    st.markdown('<div class="section-label">02 · PERIOD & CONNECTION</div>',unsafe_allow_html=True)
    if st.session_state.get("pending_period_end"):
        st.session_state["period_end"]=date.fromisoformat(st.session_state.pop("pending_period_end"))
    end=st.date_input("30-day period — end date",datetime.now().date(),key="period_end")
    auto_latest=st.checkbox("Automatically choose an available period if this period has no catalogue sources",value=True,key="auto_latest_period")
    if st.session_state.get("period_adjustment_note"):
        st.info(st.session_state["period_adjustment_note"])
    start=end-timedelta(days=29)
    st.caption(f"30 calendar days: {start} → {end}")
    cloud_warning=st.slider("Cloud/quality-masked coverage warning (%)",0,100,40)
    if st.button("🔍 Search Sentinel-2 availability",use_container_width=True):
        try:
            with st.spinner("Searching all tiles intersecting the AOI…"):
                scenes=find_catalog(ensure_aoi(st.session_state.aoi),start,end)
            st.session_state.scene_catalog={"aoi":ensure_aoi(st.session_state.aoi),"start":str(start),"end":str(end),"scenes":scenes}
        except Exception as exc:st.error(str(exc))
    catalog=st.session_state.get("scene_catalog")
    if catalog:
        if catalog["aoi"]!=ensure_aoi(st.session_state.aoi) or catalog["start"]!=str(start) or catalog["end"]!=str(end):
            st.caption("Catalogue below belongs to a previous AOI/period; search again to update availability.")
        scenes=catalog["scenes"]
        frame=pd.DataFrame([{"date":get_datetime(s)} for s in scenes])
        if not frame.empty:
            st.dataframe(frame.sort_values("date"),height=135,hide_index=True,use_container_width=True)
            st.caption(f"{len(frame)} acquisition date(s) found in the Sentinel Hub catalogue. No image/reference selection is required.")
        else:st.info("The completed catalogue search found no S2 source dates in this interval.")
    auth=st.session_state.get("cdse_auth")
    if auth:
        st.success(f"Copernicus connected · {auth.get('username','')}")
        if st.button("Log out",key="cdse_logout"):
            st.session_state.pop("cdse_auth",None);st.rerun()
    else:
        st.markdown('<div class="card-title">Copernicus login</div>',unsafe_allow_html=True)
        st.caption("Credentials are sent to the official identity service. The session keeps API tokens, not a saved password.")
        st.link_button("🌐 Open Copernicus website","https://dataspace.copernicus.eu/",use_container_width=True)
        with st.form("cdse_login_form",clear_on_submit=True):
            email=st.text_input("Copernicus email")
            password=st.text_input("Copernicus password",type="password")
            totp=st.text_input("2FA code (optional)",max_chars=8)
            submitted=st.form_submit_button("🔐 Login & connect",type="primary",use_container_width=True)
        if submitted:
            try:
                st.session_state.cdse_auth=authenticate_cdse(email,password,totp);st.rerun()
            except Exception as exc:st.error(str(exc))
st.markdown('<div class="section-label">03 · SELECT WHAT TO DISPLAY</div>',unsafe_allow_html=True)
smoke_enabled=st.checkbox("Sentinel-2 · Smoke candidates",value=True,key="show_smoke")
st.caption("Select any S5P products below. Methane and smoke have priority tabs. Both aerosol-index variants are available; cloud fields are separate context products, not pollutants.")
selected=[]
cols=st.columns(4)
for i,(key,meta) in enumerate(PRODUCTS.items()):
    with cols[i%4]:
        if st.checkbox(meta["label"],value=key in ("CH4","CO","NO2","SO2","AI340"),key="product_"+key):selected.append(key)
with st.expander("Additional Sentinel-5P cloud-context fields",expanded=False):
    context_cols=st.columns(3)
    for i,(key,meta) in enumerate(CLOUD_PRODUCTS.items()):
        with context_cols[i%3]:
            if st.checkbox(meta["label"],key="product_"+key):selected.append(key)
with st.expander("Processing settings — appearance and geographic tools are retained",expanded=False):
    p1,p2,p3=st.columns(3)
    threshold=p1.number_input("Smoke index threshold",min_value=-0.50,max_value=0.80,value=0.10,step=0.01)
    min_pixels=p2.number_input("Minimum connected candidate pixels",min_value=2,max_value=1000,value=10,step=1)
    dilation=p3.number_input("Final dilation radius",min_value=0,max_value=20,value=2,step=1)
    p4,p5,p6=st.columns(3)
    resolution=p4.selectbox("Requested S2 grid spacing (m)",[20,60,100,250,500,1000],index=3)
    brightness=p5.slider("Minimum visible brightness",0.01,0.40,0.10,0.01)
    swir_ratio=p6.slider("Maximum B12/B11 ratio",0.2,3.0,1.6,0.05)
    smooth=st.slider("Smoke-index smoothing sigma (pixels)",0.0,6.0,2.0,0.5)
    exclude_water=st.checkbox("Exclude water and shallow water from smoke candidates (recommended)",value=True)
    medium=st.checkbox("Include medium-cloud pixels in candidate screening (experimental)",value=False)
    qa_override=st.checkbox("Override official S5P QA defaults",value=False)
    qa=st.slider("S5P QA (%) when override enabled",0,100,50)
    st.caption("Official defaults: NO2 75%, other fields 50%. Lowering QA increases coverage only where low-QA retrievals exist, at a quality cost. No fabricated values are inserted.")
    st.caption("S2 context automatically uses the previous 30 days, never future observations. No image, tile or reference selection is exposed. Current-day and older candidates remain separate.")
region=shape(ensure_aoi(st.session_state.aoi))
if region.bounds[2]-region.bounds[0]>30 or region.bounds[3]-region.bounds[1]>30:
    st.warning("Please draw a regional AOI (under 30° span in longitude/latitude). This dashboard is not a global processing service.")
    st.stop()
grid=grid_for(ensure_aoi(st.session_state.aoi),resolution)
params={"threshold":float(threshold),"min_pixels":int(min_pixels),"dilation":int(dilation),
        "brightness":brightness,"swir_ratio":swir_ratio,"smooth":smooth,"medium_cloud":medium,"exclude_water":exclude_water}
config={"version":"v10.0-available","aoi":ensure_aoi(st.session_state.aoi),"start":str(start),"end":str(end),
        "products":selected,"smoke":smoke_enabled,"params":params,"qa_override":qa_override,"qa":qa,"grid":grid,
        "cloud_warning":cloud_warning}
if st.session_state.get("cdse_auth") and (smoke_enabled or selected):
    availability_key="availability_"+fingerprint(config)
    if availability_key not in st.session_state:
        with st.spinner("Checking actual source dates in the Sentinel Hub catalogue…"):
            st.session_state[availability_key]=availability_for(config)
    availability=st.session_state[availability_key]
    config["availability"]=availability
    all_success=bool(availability) and all(v["ok"] for v in availability.values())
    any_source=any(v["days"] for v in availability.values())
    if all_success and not any_source and auto_latest:
        latest_key="latest_checked_"+fingerprint([config["aoi"],str(end),selected,smoke_enabled])
        if not st.session_state.get(latest_key):
            st.session_state[latest_key]=True
            with st.spinner("No sources in the requested period; checking the latest preceding source date…"):
                latest,latest_errors=latest_available_before(config)
            if latest and latest!=end:
                st.session_state["pending_period_end"]=str(latest)
                st.session_state["period_adjustment_note"]=f"Requested period had no catalogue sources. Period end moved to actual source date {latest}; this is explicitly a different date, not fabricated data."
                st.rerun()
            elif not latest:
                st.warning("No catalogue source was found in the checked 180-day lookback. Data availability is not established; no measurements will be invented.")
                if latest_errors:st.caption(" | ".join(latest_errors))
    availability_errors={k:v["error"] for k,v in availability.items() if not v["ok"]}
    if availability_errors:
        st.warning("Availability search failed for some sources. This is a catalogue/network failure, NOT proof that imagery is absent.")
        st.json(availability_errors)
    if st.button("🔄 Refresh actual source dates",key="refresh_catalogue"):
        with st.spinner("Refreshing catalogue…"):
            st.session_state[availability_key]=availability_for(config,force=True)
        st.rerun()
else:
    availability={}
    config["availability"]={}
signature=fingerprint(config)
with st.expander("Actual source dates by product — no measurement values are inferred here",expanded=bool(availability) and not any(v["days"] for v in availability.values())):
    for key,info in availability.items():
        st.write(key," · "+(", ".join(info["days"]) if info["days"] else ("no source dates in this period" if info["ok"] else "search failed: "+str(info["error"]))))
st.caption(f"Nominal S2 regional grid spacing: {grid['res']} m · {grid['w']} × {grid['h']} pixels. Requested resolution is automatically coarsened for large AOIs; full-region detail is not claimed to be native 20 m.")
if not smoke_enabled and not selected:
    st.info("Select smoke or at least one Sentinel-5P product.")
    st.stop()
st.markdown('<div class="section-label">04 · 30-DAY EXPLORER</div>',unsafe_allow_html=True)
days=[stamp.date() for stamp in pd.date_range(start,end)]
connected=bool(st.session_state.get("cdse_auth"))
msg_slot=st.container()
left,right=st.columns([1,1])
load_one=left.button("🛰️ Display / retry selected day",type="primary",use_container_width=True,disabled=not connected)
load_month=right.button("🎬 Prepare / resume all 30 days",use_container_width=True,disabled=not connected)
st.caption("Selected-day mode downloads only what is needed. Once enabled, moving the slider fetches the new date automatically. Preparing the full month is optional, resumable, and may require many API requests.")
active_key="active_"+signature
if load_one or load_month:st.session_state[active_key]=True
if load_month:
    with msg_slot:
        progress=st.progress(0,text="Preparing 30-day archive…");status=st.empty()
        try:
            for i,one_day in enumerate(days):
                process_day(one_day,config,status)
                progress.progress(round(100*(i+1)/len(days)),text=f"{i+1}/30 days processed · {one_day}")
            lock_month_scales(config)
            summaries=[load_record(d,config) for d in days]
            errors_count=sum(len(r["errors"]) for r in summaries)
            valid_count=sum(layer.get("mean") is not None for r in summaries for layer in r["products"].values())
            if errors_count:
                st.warning(f"30 dates attempted, with {errors_count} request error(s). See the error log; resume retries failures and reuses successes.")
            else:
                st.success(f"30 dates prepared. {valid_count} product-days have valid S5P retrievals. Missing observations remain explicitly unknown.")
        except Exception as exc:st.error(f"Preparation stopped: {exc}. Click resume; completed requests are cached.")
        finally:progress.empty();status.empty()
options=[str(d) for d in days]
slider_key="shared_date_"+fingerprint([str(start),str(end),{k:v["days"] for k,v in availability.items()}])
if st.session_state.get(slider_key) not in options:
    wanted=preferred_catalogue_date(availability,end)
    st.session_state[slider_key]=wanted if wanted in options else options[-1]
def shift_day(delta):
    cur=options.index(st.session_state[slider_key])
    st.session_state[slider_key]=options[max(0,min(len(options)-1,cur+delta))]
def day_label(d):
    return d+(" ● "+"/".join(k for k,v in availability.items() if d in v["days"]) if any(d in v["days"] for v in availability.values()) else " ○")
st.markdown("**Shared date — drag the bar (or use ◀ ▶); Sentinel-2 and every selected Sentinel-5P product follow it**")
cprev,cslide,cnext=st.columns([1,12,1])
cprev.button("◀",key="prev_day_"+slider_key,on_click=shift_day,args=(-1,),use_container_width=True)
cnext.button("▶",key="next_day_"+slider_key,on_click=shift_day,args=(1,),use_container_width=True)
with cslide:
    day_string=st.select_slider("Shared date",options=options,key=slider_key,format_func=day_label,label_visibility="collapsed")
st.caption("● = catalogue source exists for the listed products on that date; ○ = no confirmed source.")
day=date.fromisoformat(day_string)
record=load_record(day,config)
needed=(smoke_enabled and "s2" not in record) or any(p not in record["products"] for p in selected)
last_request_key="last_requested_day_"+signature
should_request=load_one or (needed and st.session_state.get(last_request_key)!=day_string)
if connected and st.session_state.get(active_key) and should_request:
    st.session_state[last_request_key]=day_string
    status=st.empty()
    try:
        with st.spinner(f"Preparing {day}…"):
            record=process_day(day,config,status,retry=load_one)
    except Exception as exc:st.error(str(exc))
    finally:status.empty()

def tl_ok(kind,d):
    rec=load_record(d,config)
    if kind=="S2":
        return bool(rec.get("s2")) and rec["s2"].get("observed_pixels",0)>0
    layer=rec["products"].get(kind)
    return bool(layer) and bool(layer.get("path")) and layer.get("mean") is not None

def tl_frame(kind,d):
    rec=load_record(d,config)
    if kind=="S2":
        item=rec["s2"]
        bands,profile,inside=read_s2(item["path"],config["aoi"])
        res=smoke_screen(bands,d,config["params"])
        base=rgb_rgba(bands,inside).copy()
        base[res["current"]]=[230,57,70,255]
        label=f"{d} · Sentinel-2 · red = same-day candidates"
    else:
        layer=rec["products"][kind]
        values,profile,inside=read_s5(layer["path"],config["aoi"])
        base=field_rgba(values,fixed_limits(kind,config),ALL_PRODUCTS[kind]["cmap"],255)
        label=f"{d} · {ALL_PRODUCTS[kind]['label']}"
    alpha=base[:,:,3:4].astype(np.float32)/255
    rgb=base[:,:,:3].astype(np.float32)*alpha+np.array([226,233,236],dtype=np.float32)*(1-alpha)
    img=Image.fromarray(rgb.astype(np.uint8))
    scale=max(1,int(round(760/max(1,img.width))))
    img=img.resize((img.width*scale,img.height*scale),Image.NEAREST)
    from PIL import ImageDraw
    draw=ImageDraw.Draw(img);draw.rectangle([0,0,min(img.width,len(label)*7+12),20],fill=(255,255,255))
    draw.text((6,4),label,fill=(17,17,17))
    return img

kinds=(["S2"] if smoke_enabled else [])+list(selected)
with st.expander("🎞 Timeline player — drag the bar or press Play to watch the AOI change day by day",expanded=True):
    if not kinds:
        st.info("Select a layer first.")
    else:
        tl_kind=st.selectbox("Layer",kinds,key="tl_kind_"+signature,
            format_func=lambda k:"Sentinel-2 · smoke candidates (red)" if k=="S2" else ALL_PRODUCTS[k]["label"])
        avail=[str(d) for d in days if tl_ok(tl_kind,d)]
        if len(avail)<2:
            st.info("Prepare all 30 days first. At least two days with valid data are needed for the player.")
        else:
            tl_key="tl_frame_"+signature+tl_kind
            if st.session_state.get(tl_key) not in avail:st.session_state[tl_key]=avail[-1]
            tl_day=st.select_slider("Day",options=avail,key=tl_key)
            speed=st.slider("Play speed (seconds per frame)",0.1,2.0,0.5,0.1,key="tl_speed_"+signature)
            play=st.button("▶ Play from selected day",key="tl_play_"+signature)
            frame_slot=st.empty()
            if tl_kind!="S2":
                st.image(colorbar_png(*fixed_limits(tl_kind,config),ALL_PRODUCTS[tl_kind]["cmap"],ALL_PRODUCTS[tl_kind]["unit"]),use_container_width=True)
                st.caption("Same fixed colour scale on every day, so changes between frames are real changes. Missing pixels stay grey.")
            if play:
                for one in avail[avail.index(tl_day):]:
                    frame_slot.image(tl_frame(tl_kind,date.fromisoformat(one)),use_container_width=True)
                    time.sleep(speed)
            else:
                frame_slot.image(tl_frame(tl_kind,date.fromisoformat(tl_day)),use_container_width=True)
if not record.get("s2") and not record["products"]:
    if record["errors"]:
        st.error("No layer completed for this date. Exact errors are shown below.")
    else:st.info("Connect Copernicus, select products, then display the selected day. No image or reference needs selecting.")
else:
    st.markdown(f'<div class="result-head">{day} · Whole-AOI results</div><div class="result-sub">One geographic extent. All product maps use the selected AOI; older S2 context is explicitly dated.</div>',unsafe_allow_html=True)
    tabspec=[]
    if smoke_enabled:tabspec.append(("S2","🌫 Smoke · Sentinel-2"))
    if "CH4" in selected:tabspec.append(("CH4","🛰 Methane · CH4"))
    for product in selected:
        if product!="CH4":tabspec.append((product,ALL_PRODUCTS[product]["label"]))
    tabs=st.tabs([title for _,title in tabspec])
    for tab,(product,title) in zip(tabs,tabspec):
        with tab:
            if product=="S2":
                item=record.get("s2")
                if not item:
                    st.warning("S2 request/catalogue did not complete successfully. Read the exact error below; this is NOT a measured absence of smoke.")
                    if record["errors"].get("S2"):st.error(record["errors"]["S2"])
                    continue
                if item.get("observed_pixels",0)==0:
                    st.warning("No valid S2 pixels were returned from the attempted acquisition dates. An empty map is not presented as a smoke result.")
                    st.json(item.get("context",{}))
                    continue
                bands,profile,inside=read_s2(item["path"],config["aoi"])
                result=smoke_screen(bands,day,params)
                m1,m2,m3,m4=st.columns(4)
                m1.metric("30-day image-context coverage",f"{item['coverage']:.1f}%")
                m2.metric("Same-day assessable coverage",f"{item['current_coverage']:.1f}%")
                m3.metric("Same-day candidate area",f"{item['current_area']:.2f} km²" if item["current_area"] is not None else "No same-day data")
                m4.metric("Older-context candidate area",finite_text(item.get("historical_area"))+" km²")
                st.caption("Red = same-day spectral candidates. Amber = candidates on older observations, NOT smoke confirmed on the selected day. Map background is geographic context, not invented satellite imagery.")
                if item["coverage"]>0 and 100*(1-item["quality_coverage"]/item["coverage"])>cloud_warning:
                    st.warning("Cloud/quality-masked fraction exceeds the selected warning threshold; those pixels are not assumed smoke-free.")
                if item["coverage"]<99:
                    st.warning(f"Even the automatic 30-day context has only {item['coverage']:.1f}% observed coverage. Remaining locations have no returned S2 observation; changing a reference cannot create those measurements.")
                if item["current_coverage"]<99:
                    st.info("The image is a last-known-observation mosaic, not a single-date photograph. Older pixels improve geographic context only; current-day smoke metrics exclude them.")
                show_history=st.checkbox("Show older-observation candidate context in amber",value=False,key="history_"+signature)
                fmap=new_result_map(config["aoi"])
                rgba=rgb_rgba(bands,inside)
                show_rgb=st.checkbox("Show S2 RGB background (optional)",value=False,key="rgb_"+signature)
                add_raster(fmap,rgba,profile,"S2 actual observation context (optional)",show=show_rgb)
                unknown=inside & ~result["observed"]
                hatch=np.zeros((*unknown.shape,4),dtype=np.uint8)
                rr,cc=np.indices(unknown.shape)
                hatched=unknown & (((rr+cc)%18)<3)
                hatch[hatched]=[69,123,157,85]
                add_raster(fmap,hatch,profile,"No S2 observation — geographic context only")
                if show_history:add_raster(fmap,mask_rgba(result["historical"],(245,158,11)),profile,"OLDER spectral candidates (not current-day smoke)")
                add_raster(fmap,mask_rgba(result["current"],(230,57,70)),profile,"SAME-DAY spectral candidates")
                ages=field_rgba(result["age"],[0,29],"viridis_r",200)
                add_raster(fmap,ages,profile,"Observation age (0–29 days)",show=False)
                add_raster(fmap,field_rgba(np.where(result["observed"],result["index"],np.nan),[-1,1],"inferno"),profile,"Smoke index (not a validated smoke map)",show=False)
                finish_map(fmap,config["aoi"],"smoke_map_"+signature+str(day),height=620)
                st.markdown('<span class="result-chip">🔴 Same-day candidate</span><span class="result-chip">🟠 Older candidate</span><span class="result-chip">▧ No S2 observation</span>',unsafe_allow_html=True)
                with st.expander("Observation age and smoke-screen diagnostics"):
                    st.json(item.get("context",{}))
                    st.image(colorbar_png(0,29,"viridis_r","Observation age · days"),use_container_width=True)
                    st.caption(f"Nominal grid spacing: {grid['res']} m. Index=(B03−B11)/(B03+B11). Water is not globally excluded. Clouds, haze, glint and bright surfaces can create false positives; dark smoke can be missed.")
                    st.caption(f"Same-day assessed pixels: {item['current_assessable_pixels']:,}; same-day candidate pixels: {item['current_pixels']:,}.")
                if item.get("candidate_fraction",0)>0.50:
                    st.warning("More than half of assessable context pixels were flagged: probable water/haze/cloud/surface contamination. This is not confirmed smoke coverage.")
                st.warning("Exploratory smoke-candidate screening, not a validated smoke classifier or proof of source. These masks cannot establish conflict attribution.")
                with st.expander("Download Sentinel-2 outputs"):
                    candidate=np.where(result["fresh"],result["current"].astype(np.float32),np.nan)
                    historical=np.where(result["quality"] & ~result["fresh"],result["historical"].astype(np.float32),np.nan)
                    c1,c2=st.columns(2)
                    c1.download_button("⬇ Same-day mask GeoTIFF",tif_bytes(candidate,profile),file_name=f"{day}_same_day_candidates.tif",mime="image/tiff",key="s2_same_tif")
                    c2.download_button("⬇ Older-context mask GeoTIFF",tif_bytes(historical,profile),file_name=f"{day}_context_candidates.tif",mime="image/tiff",key="s2_history_tif")
                    c1.download_button("⬇ Observation age GeoTIFF",tif_bytes(result["age"],profile),file_name=f"{day}_observation_age.tif",mime="image/tiff",key="s2_age_tif")
                    c1.download_button("⬇ Acquisition date GeoTIFF (days since 1970-01-01)",tif_bytes(bands["observation_day"],profile),file_name=f"{day}_acquisition_date.tif",mime="image/tiff",key="s2_date_tif")
                    c2.download_button("⬇ Smoke index GeoTIFF",tif_bytes(np.where(result["observed"],result["index"],np.nan),profile),file_name=f"{day}_smoke_index.tif",mime="image/tiff",key="s2_index_tif")
                    c1.download_button("⬇ Validity mask GeoTIFF",tif_bytes(np.where(inside,result["quality"].astype(np.float32),np.nan),profile),file_name=f"{day}_validity.tif",mime="image/tiff",key="s2_validity")
                    c2.download_button("⬇ Georeferenced RGB PNG package",package_png(rgba,profile),file_name=f"{day}_rgb_georeferenced.zip",mime="application/zip",key="s2_png_zip")
            else:
                meta=ALL_PRODUCTS[product];layer=record["products"].get(product)
                if not layer:
                    st.warning("Request did not complete. Unknown is not zero.")
                    if record["errors"].get(product):st.caption(record["errors"][product])
                    continue
                if not layer.get("path"):
                    st.info("No catalogue acquisition for this product on the selected day. Choose a date marked for this product on the shared slider.")
                    st.caption("Available source dates: "+", ".join(layer.get("available_days",[])))
                    continue
                explore=False
                probe=layer.get("qa0_probe",{})
                if probe.get("valid_cells",0)>0:
                    st.warning("Quality-filtered retrieval is empty, but a QA=0 diagnostic contains data. This is a quality-filtering issue, not absence of all measurements.")
                    explore=st.checkbox("Show QA=0 exploratory data (not part of the quality-filtered time series)",value=False,key="explore_"+product+signature)
                    if explore:
                        layer={**layer,**probe}
                        st.warning("EXPLORATORY QA=0 DISPLAY: low-quality retrievals are included. Do not interpret this as the standard quality-filtered product.")
                values,profile,inside=read_s5(layer["path"],config["aoi"])
                m1,m2,m3=st.columns(3)
                m1.metric("Direct-product display-grid mean",finite_text(layer.get("mean"))+(" "+meta["unit"] if layer.get("mean") is not None else ""))
                m2.metric("Valid AOI coverage",f"{layer['coverage']:.1f}%")
                qa_used=0 if explore else (qa if qa_override else (75 if product=="NO2" else 50))
                m3.metric("Quality threshold",f"QA ≥ {qa_used}%")
                if product=="CH4":st.info("CH4 is the column-averaged dry-air methane mixing ratio (XCH4), not a surface concentration or emission rate. Data gaps, including over water, remain unknown.")
                st.caption(f"Calendar-day retrievals: {day}, UTC. Actual acquisition times need not match S2. No spatial or temporal gap filling is used for S5P measurements.")
                if layer["mean"] is None:
                    st.warning("A catalogue source exists, but Process API returned no valid pixels at this AOI/date/QA. No empty basemap is shown as a pollution result.")
                    st.json({k:v for k,v in layer.items() if k!="path"})
                    st.download_button("⬇ Original API TIFF for inspection",Path(layer["path"]).read_bytes(),file_name=f"{day}_{product}_api.tif",mime="image/tiff",key="raw_empty_"+product)
                    continue
                limits=fixed_limits(product,config)
                rgba=field_rgba(values,limits,meta["cmap"])
                fmap=new_result_map(config["aoi"])
                add_raster(fmap,rgba,profile,title)
                finish_map(fmap,config["aoi"],"s5_map_"+signature+product+str(day),height=560)
                st.image(colorbar_png(*limits,meta["cmap"],meta["unit"]),use_container_width=True)
                st.caption(f"Frozen display colour range: {limits[0]:.5g} to {limits[1]:.5g} {meta['unit']}. Outside-range values saturate visually but remain unchanged in GeoTIFF. Basemap gaps do not mean clean air.")
                c1,c2=st.columns(2)
                c1.download_button("⬇ GeoTIFF",tif_bytes(values,profile),file_name=f"{day}_{product}.tif",mime="image/tiff",key="tif_"+product)
                c2.download_button("⬇ Georeferenced PNG package",package_png(rgba,profile),file_name=f"{day}_{product}_png.zip",mime="application/zip",key="png_"+product)
    st.markdown("### Selected-product overview")
    if record["products"]:
        overview=[]
        for p in selected:
            layer=record["products"].get(p,{})
            overview.append({"Product":ALL_PRODUCTS[p]["label"],"Mean":finite_text(layer.get("mean")),"Unit":ALL_PRODUCTS[p]["unit"],
                             "Valid coverage (%)":layer.get("coverage"),"Status":record["errors"].get(p,layer.get("status","not requested"))})
        st.dataframe(pd.DataFrame(overview),use_container_width=True,hide_index=True)
    if st.checkbox("Show all selected atmospheric products on one comparison map",key="compare_map_"+signature):
        fmap=new_result_map(config["aoi"])
        for product in selected:
            layer=record["products"].get(product)
            if layer and layer.get("path") and layer.get("mean") is not None:
                values,p,_=read_s5(layer["path"],config["aoi"])
                add_raster(fmap,field_rgba(values,fixed_limits(product,config),ALL_PRODUCTS[product]["cmap"]),p,
                    ALL_PRODUCTS[product]["label"],show=product==selected[0])
        finish_map(fmap,config["aoi"],"compare_"+signature+str(day),height=550)
        st.caption("Enable individual layers in the map control. Each product has its own unit and fixed colour scale; overlapping colours are not a combined pollution index.")

with st.expander("Download catalogue + request diagnostic report"):
    st.download_button("⬇ Diagnostic JSON",json.dumps({"app_version":"v10.0-available","config":config,"day_record":record},ensure_ascii=False,indent=2,default=str),
        file_name=f"{day}_catalogue_diagnostics.json",mime="application/json",key="diagnostic_json")

st.markdown('<div class="section-label">05 · DAILY CURVES & DOWNLOADS</div>',unsafe_allow_html=True)
df=table_for(config)
if df.select_dtypes(include="number").notna().any().any():
    if smoke_enabled:
        st.caption("S2 same-day candidate area (km²) — assess coverage before comparing days. Older-context candidates are not plotted as current smoke.")
        st.line_chart(df[["S2_current_candidate_km2"]],height=190)
        st.line_chart(df[["S2_current_coverage_pct","S2_context_coverage_pct"]],height=170)
    for product in selected:
        st.caption(f"{ALL_PRODUCTS[product]['label']} · {ALL_PRODUCTS[product]['unit']}")
        if df[product].notna().any():st.line_chart(df[[product]],height=180)
        else:st.info(f"{product}: no valid quality-filtered daily values have been downloaded. No zero series was created.")
    with st.expander("Daily values, coverage and missing observations"):
        st.dataframe(readable_table(df),use_container_width=True)
    st.download_button("⬇ 30-day observations & coverage CSV",df.to_csv(),file_name="daily_observations_coverage.csv",mime="text/csv",key="month_csv")
    st.caption("Not-yet-requested days and missing observations are NaN, never zero. Use Prepare / resume all 30 days for the full archive. Curves use display-grid means, not an emission inventory.")
errors=[]
for d in days:
    r=load_record(d,config)
    for product,error in r["errors"].items():errors.append({"date":str(d),"product":product,"error":error})
if errors:
    with st.expander(f"Exact request errors ({len(errors)})",expanded=True):
        error_df=pd.DataFrame(errors);st.dataframe(error_df,use_container_width=True)
        st.download_button("⬇ Error log",error_df.to_csv(index=False),file_name="request_errors.csv",mime="text/csv")
if st.button("📦 Prepare downloaded 30-day archive",disabled=not df.select_dtypes(include="number").notna().any().any()):
    archive_path=session_folder(config)/"downloaded_archive.zip"
    with st.spinner("Packaging cached rasters…"):
        with zipfile.ZipFile(archive_path,"w",zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("config.json",json.dumps(config,ensure_ascii=False,indent=2))
            archive.writestr("daily_observations_coverage.csv",df.to_csv())
            archive.writestr("request_errors.json",json.dumps(errors,ensure_ascii=False))
            archive.writestr("README.txt","S2 TIFF band order: B02,B03,B04,B08,B11,B12,SCL,dataMask,observation_day. observation_day is integer days since 1970-01-01. S2 is past-only last-known context, not a same-day photograph. S5P bands: direct daily value,dataMask,valid indicator. Catalogue acquisition dates are checked first. Missing retrievals are not zero. All layers are exploratory; no source attribution. config.json defines AOI, QA, grid and screening settings. S2 is a spectral screen, not a validated smoke classifier.")
            for d in days:
                r=load_record(d,config)
                archive.writestr(str(d)+"/status.json",json.dumps(r,ensure_ascii=False,indent=2))
                if r.get("s2"):archive.write(r["s2"]["path"],arcname=str(d)+"/S2_observation_context.tif")
                for product,layer in r["products"].items():
                    if layer.get("path"):archive.write(layer["path"],arcname=str(d)+"/"+product+".tif")
    st.session_state["archive_"+signature]=str(archive_path)
archive_path=st.session_state.get("archive_"+signature)
if archive_path and Path(archive_path).exists():
    st.download_button("⬇ Download prepared archive",Path(archive_path).read_bytes(),file_name="smoke_methane_30days.zip",mime="application/zip",key="archive_download")
