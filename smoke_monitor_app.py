"""AOI-first Sentinel smoke / methane dashboard.
No reference-image selector. Daily S5P retrievals; automatic past-only S2 context.
S2 composite pixels carry their acquisition date. Stale candidates are not daily smoke.
"""
from __future__ import annotations
import io, json, math, re, hashlib, time, zipfile, base64
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Optional
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
TOKEN_URL="https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
PROCESS_URL="https://sh.dataspace.copernicus.eu/api/v1/process"
CACHE_DIR=Path.home()/".smoke_dashboard_v8"
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
    g=shape(ensure_aoi(aoi)); c=g.centroid
    zone=max(1,min(60,int((c.x+180)//6)+1))
    epsg=(32600 if c.y>=0 else 32700)+zone
    xmin,ymin,xmax,ymax=transform_bounds("EPSG:4326",f"EPSG:{epsg}",*g.bounds,densify_pts=21)
    res=max(resolution,math.ceil(max(xmax-xmin,ymax-ymin)/max_side))
    w=max(1,math.ceil((xmax-xmin)/res)); h=max(1,math.ceil((ymax-ymin)/res))
    return {"bounds":[xmin,ymax-h*res,xmin+w*res,ymax],"epsg":epsg,"w":w,"h":h,"res":res}


def response_error(response):
    try:
        e=response.json().get("error",{})
        return e.get("message",str(e)) if isinstance(e,dict) else str(e)
    except Exception:
        return response.text[:1200]


def process_request(payload,key,bands):
    path=CACHE_DIR/(key+".tif")
    if path.exists():
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


def s2_script(day):
    target=datetime.combine(day,datetime.min.time())+timedelta(hours=12)
    return """//VERSION=3
function setup(){return {
 input:[{bands:["B02","B03","B04","B08","B11","B12","SCL","dataMask"],
 units:["REFLECTANCE","REFLECTANCE","REFLECTANCE","REFLECTANCE","REFLECTANCE","REFLECTANCE","DN","DN"]}],
 mosaicking:"TILE",output:{bands:9,sampleType:"FLOAT32"}};}
function evaluatePixel(samples,scenes){
 var best=null,bestScore=1e30,obs=-9999;
 var target=Date.parse("__TARGET__");
 for(var i=0;i<samples.length;i++){
  var s=samples[i]; if(s.dataMask!==1)continue;
  var stamp=Date.parse(scenes.tiles[i].date);if(!isFinite(stamp))continue;
  var age=(target-stamp)/86400000;
  var penalty=([0,1,3,8,9,10,11].indexOf(s.SCL)>=0)?1000:0;
  var score=penalty+Math.max(0,age);
  if(score<bestScore){best=s;bestScore=score;obs=Math.floor(stamp/86400000);}
 }
 if(best===null)return [0,0,0,0,0,0,0,0,-9999];
 return [best.B02,best.B03,best.B04,best.B08,best.B11,best.B12,best.SCL,1,obs];
}
""".replace("__TARGET__",target.strftime("%Y-%m-%dT%H:%M:%SZ"))


def request_s2(day,config):
    # No user-selected scene, tile or reference: all sources in the past 30 days.
    g=config["grid"]
    start=day-timedelta(days=29); end=day+timedelta(days=1)
    payload={"input":{"bounds":{"bbox":g["bounds"],"properties":{"crs":f"http://www.opengis.net/def/crs/EPSG/0/{g['epsg']}"}},
      "data":[{"type":"sentinel-2-l2a","dataFilter":{
        "timeRange":{"from":str(start)+"T00:00:00Z","to":str(end)+"T00:00:00Z"},
        "mosaickingOrder":"mostRecent","maxCloudCoverage":100},
        "processing":{"upsampling":"NEAREST","downsampling":"NEAREST"}}]},
      "output":{"width":g["w"],"height":g["h"],"responses":[{"identifier":"default","format":{"type":"image/tiff"}}]},
      "evalscript":s2_script(day)}
    return process_request(payload,fingerprint(["s2-past30-v8",payload]),9)


def read_s2(path,aoi):
    with rasterio.open(path) as ds:
        array=ds.read().astype(np.float32); profile=ds.profile.copy()
    inside=clip_mask(profile,*array.shape[1:],aoi)
    observed=(array[7]>0.5)&inside
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
    candidate=quality & (index>params["threshold"]) & (bright>params["brightness"]) & (ratio<params["swir_ratio"]) & (vegetation<0.45)
    if params["smooth"]>0:
        weights=gaussian_filter(quality.astype(np.float32),sigma=params["smooth"])
        values=gaussian_filter(np.where(quality,index,0),sigma=params["smooth"])
        smooth=np.divide(values,weights,out=np.full_like(values,np.nan),where=weights>0.05)
        candidate &= smooth>params["threshold"]
    components,count=label(candidate,structure=np.ones((3,3)))
    sizes=np.bincount(components.ravel(),minlength=count+1)
    keep=np.flatnonzero(sizes>=params["min_pixels"]);keep=keep[keep>0]
    smoke=np.isin(components,keep)&quality
    if params["dilation"] and smoke.any():
        smoke=binary_dilation(smoke,structure=disk(params["dilation"]))&quality
    age=(day-EPOCH).days-bands["observation_day"]
    fresh=quality & (age>=0) & (age<1)
    return {"observed":observed,"quality":quality,"smoke":smoke,"fresh":fresh,
            "current":smoke&fresh,"historical":smoke&~fresh,"age":np.where(observed,age,np.nan),
            "index":index}


def s5_script(band):
    # Every output pixel is the mean of valid observations on this calendar day.
    return """//VERSION=3
function setup(){return {input:[{bands:["__BAND__","dataMask"]}],
 mosaicking:"ORBIT",output:{bands:3,sampleType:"FLOAT32"}};}
function evaluatePixel(samples){var total=0,n=0;
 for(var i=0;i<samples.length;i++){var s=samples[i];
 if(s.dataMask===1&&isFinite(s.__BAND__)){total+=s.__BAND__;n++;}}
 return n>0?[total/n,1,n]:[0,0,0];}
""".replace("__BAND__",band)


def request_s5(day,product,config):
    bounds=shape(config["aoi"]).bounds
    ratio=max(0.1,(bounds[2]-bounds[0])*math.cos(math.radians((bounds[1]+bounds[3])/2))/max(1e-6,bounds[3]-bounds[1]))
    w=max(1,round(420*min(1,ratio)));h=max(1,round(420*min(1,1/ratio)))
    qa=config["qa"] if config["qa_override"] else (75 if product=="NO2" else 50)
    payload={"input":{"bounds":{"bbox":list(bounds),"properties":{"crs":"http://www.opengis.net/def/crs/EPSG/0/4326"}},
        "data":[{"type":"sentinel-5p-l2","dataFilter":{"timeRange":{
            "from":str(day)+"T00:00:00Z","to":str(day+timedelta(days=1))+"T00:00:00Z"},
            "mosaickingOrder":"mostRecent"},"processing":{"minQa":qa,"upsampling":"NEAREST"}}]},
        "output":{"width":w,"height":h,"responses":[{"identifier":"default","format":{"type":"image/tiff"}}]},
        "evalscript":s5_script(ALL_PRODUCTS[product]["band"])}
    return process_request(payload,fingerprint(["s5-daily-v8",payload]),3)


def read_s5(path,aoi):
    with rasterio.open(path) as ds:
        values=ds.read(1).astype(np.float32); valid=ds.read(2)>0.5
        profile=ds.profile.copy()
    inside=clip_mask(profile,*values.shape,aoi)
    values[~(valid&inside)|~np.isfinite(values)]=np.nan
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
    folium.TileLayer("CartoDB positron",name="Geographic context (not satellite evidence)").add_to(fmap)
    folium.TileLayer("OpenStreetMap",name="OpenStreetMap context",show=False).add_to(fmap)
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
    if config["smoke"] and ("s2" not in record or record["errors"].get("S2")):
        if status:status.caption(f"{day} · S2 automatic AOI mosaic (last 30 days; acquisition ages retained)")
        try:
            path=request_s2(day,config)
            bands,profile,inside=read_s2(path,config["aoi"])
            result=smoke_screen(bands,day,config["params"])
            area=abs(profile["transform"].a*profile["transform"].e)/1e6
            denom=max(1,int(inside.sum()))
            record["s2"]={"path":str(path),"coverage":float(result["observed"].sum()/denom*100),
                "quality_coverage":float(result["quality"].sum()/denom*100),
                "current_coverage":float(result["fresh"].sum()/denom*100),
                "current_area":float(result["current"].sum()*area) if result["fresh"].any() else None,
                "historical_area":float(result["historical"].sum()*area),
                "current_pixels":int(result["current"].sum()),"observed_pixels":int(result["observed"].sum()),
                "current_assessable_pixels":int(result["fresh"].sum()),
                "candidate_fraction":float(result["smoke"].sum()/max(1,result["quality"].sum()))}
            record["errors"].pop("S2",None)
        except Exception as exc:record["errors"]["S2"]=str(exc)
    for product in config["products"]:
        if product in record["products"] and not record["errors"].get(product):continue
        if status:status.caption(f"{day} · {ALL_PRODUCTS[product]['label']} · daily retrieval")
        try:
            path=request_s5(day,product,config)
            values,profile,inside=read_s5(path,config["aoi"])
            finite=values[np.isfinite(values)]
            record["products"][product]={"path":str(path),"coverage":float(finite.size/max(1,inside.sum())*100),
                "mean":float(finite.mean()) if finite.size else None,
                "min":float(finite.min()) if finite.size else None,"max":float(finite.max()) if finite.size else None}
            record["errors"].pop(product,None)
        except Exception as exc:record["errors"][product]=str(exc)
        atomic_json(session_folder(config)/(str(day)+".json"),record)
    atomic_json(session_folder(config)/(str(day)+".json"),record)
    return record


def table_for(config):
    rows=[]
    for stamp in pd.date_range(config["start"],config["end"]):
        d=stamp.date(); record=load_record(d,config)
        row={"date":stamp}
        s2=record.get("s2",{})
        row["S2_current_candidate_km2"]=s2.get("current_area",np.nan)
        row["S2_current_coverage_pct"]=s2.get("current_coverage",np.nan)
        row["S2_context_coverage_pct"]=s2.get("coverage",np.nan)
        for product in config["products"]:
            p=record["products"].get(product,{})
            row[product]=p.get("mean",np.nan)
            row[product+"_coverage_pct"]=p.get("coverage",np.nan)
        rows.append(row)
    return pd.DataFrame(rows).set_index("date")


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
            record=load_record(stamp.date(),config)
            layer=record["products"].get(product)
            if layer:
                values,_,_=read_s5(layer["path"],config["aoi"])
                finite=values[np.isfinite(values)]
                if finite.size:chunks.append(finite[::max(1,finite.size//1500)])
        if chunks:
            lo,hi=np.percentile(np.concatenate(chunks),[2,98])
            if hi<=lo:
                delta=max(abs(float(lo))*0.02,1e-9);lo-=delta;hi+=delta
            saved[product]=[float(lo),float(hi)]
    atomic_json(session_folder(config)/"scales.json",saved)


def find_catalog(aoi,start,end):
    url=STAC_URL+"search"
    payload={"collections":["sentinel-2-l2a"],"datetime":str(start)+"T00:00:00Z/"+str(end)+"T23:59:59Z",
             "intersects":aoi,"limit":100}
    features=[];seen=set();method="POST"
    from urllib.parse import urljoin
    for _ in range(150):
        response=requests.post(url,json=payload,timeout=(25,90)) if method=="POST" else requests.get(url,timeout=(25,90))
        response.raise_for_status();data=response.json();added=0
        for item in data.get("features",[]):
            if item.get("id") not in seen:
                seen.add(item.get("id"));features.append(item);added+=1
        nxt=next((link for link in data.get("links",[]) if link.get("rel")=="next"),None)
        if not nxt:return features
        if not added:raise RuntimeError("Repeated STAC page. Shorten the search period.")
        url=urljoin(url,nxt["href"]);method=nxt.get("method","GET").upper()
        if method=="POST":
            body=nxt.get("body",{});payload={**payload,**body} if nxt.get("merge") else (body or payload)
    raise RuntimeError("STAC pagination limit reached; no incomplete list was accepted.")


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
                       height=365,use_container_width=True,key="aoi_map")
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
    end=st.date_input("30-day period — end date",datetime.now().date(),key="period_end")
    start=end-timedelta(days=29)
    st.caption(f"30 calendar days: {start} → {end}")
    cloud_warning=st.slider("Scene cloud metadata warning (%)",0,100,40)
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
        frame=pd.DataFrame([{"date":get_datetime(s),"tile":get_tile(s),"cloud":get_cloud(s)} for s in scenes])
        if not frame.empty:
            st.dataframe(frame.sort_values("date"),height=135,hide_index=True,use_container_width=True)
            st.caption(f"{len(frame)} tile records; {(frame['cloud']>cloud_warning).sum()} above warning threshold. No tiles are discarded by this warning.")
        else:st.info("No catalogue records in that search interval.")
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
smoke_enabled=st.checkbox("Sentinel-2 · Smoke and RGB context",value=True,key="show_smoke")
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
        "brightness":brightness,"swir_ratio":swir_ratio,"smooth":smooth,"medium_cloud":medium}
config={"version":"v8.1","aoi":ensure_aoi(st.session_state.aoi),"start":str(start),"end":str(end),
        "products":selected,"smoke":smoke_enabled,"params":params,"qa_override":qa_override,"qa":qa,"grid":grid}
signature=fingerprint(config)
st.caption(f"Actual S2 regional grid: {grid['res']} m · {grid['w']} × {grid['h']} pixels. Requested resolution is automatically coarsened for large AOIs; full-region detail is not claimed to be native 20 m.")
if not smoke_enabled and not selected:
    st.info("Select smoke or at least one Sentinel-5P product.")
    st.stop()
st.markdown('<div class="section-label">04 · 30-DAY EXPLORER</div>',unsafe_allow_html=True)
days=[stamp.date() for stamp in pd.date_range(start,end)]
# The slider exists before preparation and remains the only date selector for results.
day_string=st.select_slider("Shared date — Sentinel-2 and every selected Sentinel-5P product",
    options=[str(day) for day in days],value=str(end),key="shared_date_"+fingerprint([str(start),str(end)]))
day=date.fromisoformat(day_string)
left,right=st.columns([1,1])
connected=bool(st.session_state.get("cdse_auth"))
load_one=left.button("🛰️ Display / retry selected day",type="primary",use_container_width=True,disabled=not connected)
load_month=right.button("🎬 Prepare / resume all 30 days",use_container_width=True,disabled=not connected)
st.caption("Selected-day mode downloads only what is needed. Once enabled, moving the slider fetches the new date automatically. Preparing the full month is optional, resumable, and may require many API requests.")
active_key="active_"+signature
if load_one or load_month:st.session_state[active_key]=True
if load_month:
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
                    st.warning("S2 observation unavailable for this date. No pollution value has been inferred.")
                    if record["errors"].get("S2"):st.caption(record["errors"]["S2"])
                    continue
                bands,profile,inside=read_s2(item["path"],config["aoi"])
                result=smoke_screen(bands,day,params)
                m1,m2,m3,m4=st.columns(4)
                m1.metric("30-day image-context coverage",f"{item['coverage']:.1f}%")
                m2.metric("Same-day assessable coverage",f"{item['current_coverage']:.1f}%")
                m3.metric("Same-day candidate area",f"{item['current_area']:.2f} km²" if item["current_area"] is not None else "No same-day data")
                m4.metric("Older-context candidate area",f"{item['historical_area']:.2f} km²")
                st.caption("Red = same-day spectral candidates. Amber = candidates on older observations, NOT smoke confirmed on the selected day. Map background is geographic context, not invented satellite imagery.")
                if item["coverage"]<99:
                    st.warning(f"Even the automatic 30-day context has only {item['coverage']:.1f}% observed coverage. Remaining locations have no returned S2 observation; changing a reference cannot create those measurements.")
                if item["current_coverage"]<99:
                    st.info("The image is a last-known-observation mosaic, not a single-date photograph. Older pixels improve geographic context only; current-day smoke metrics exclude them.")
                show_history=st.checkbox("Show older-observation candidate context in amber",value=True,key="history_"+signature)
                fmap=new_result_map(config["aoi"])
                rgba=rgb_rgba(bands,inside)
                add_raster(fmap,rgba,profile,"S2 last-known RGB (past 30 days)")
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
                    st.image(colorbar_png(0,29,"viridis_r","Observation age · days"),use_container_width=True)
                    st.caption(f"Grid: {grid['res']} m. Index=(B03−B11)/(B03+B11). Water is not globally excluded. Clouds, haze, glint and bright surfaces can create false positives; dark smoke can be missed.")
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
                values,profile,inside=read_s5(layer["path"],config["aoi"])
                m1,m2,m3=st.columns(3)
                m1.metric("AOI display-grid mean",f"{layer['mean']:.5g} {meta['unit']}" if layer["mean"] is not None else "No valid retrieval")
                m2.metric("Valid AOI coverage",f"{layer['coverage']:.1f}%")
                qa_used=qa if qa_override else (75 if product=="NO2" else 50)
                m3.metric("Quality threshold",f"QA ≥ {qa_used}%")
                if product=="CH4":st.info("CH4 is the column-averaged dry-air methane mixing ratio (XCH4), not a surface concentration or emission rate. Data gaps, including over water, remain unknown.")
                st.caption(f"Calendar-day retrievals: {day}, UTC. Actual acquisition times need not match S2. No spatial or temporal gap filling is used for S5P measurements.")
                if layer["mean"] is None:st.warning("No valid retrievals for this product/day/AOI/QA setting. The map below remains geographic context only.")
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
            overview.append({"Product":ALL_PRODUCTS[p]["label"],"Mean":layer.get("mean"),"Unit":ALL_PRODUCTS[p]["unit"],
                             "Valid coverage (%)":layer.get("coverage"),"Status":record["errors"].get(p,"valid" if layer.get("mean") is not None else "no valid retrieval")})
        st.dataframe(pd.DataFrame(overview),use_container_width=True,hide_index=True)
    if st.checkbox("Show all selected atmospheric products on one comparison map",key="compare_map_"+signature):
        fmap=new_result_map(config["aoi"])
        for product in selected:
            layer=record["products"].get(product)
            if layer:
                values,p,_=read_s5(layer["path"],config["aoi"])
                add_raster(fmap,field_rgba(values,fixed_limits(product,config),ALL_PRODUCTS[product]["cmap"]),p,
                    ALL_PRODUCTS[product]["label"],show=product==selected[0])
        finish_map(fmap,config["aoi"],"compare_"+signature+str(day),height=550)
        st.caption("Enable individual layers in the map control. Each product has its own unit and fixed colour scale; overlapping colours are not a combined pollution index.")

st.markdown('<div class="section-label">05 · DAILY CURVES & DOWNLOADS</div>',unsafe_allow_html=True)
df=table_for(config)
if df.notna().any().any():
    if smoke_enabled:
        st.caption("S2 same-day candidate area (km²) — assess coverage before comparing days. Older-context candidates are not plotted as current smoke.")
        st.line_chart(df[["S2_current_candidate_km2"]],height=190)
        st.line_chart(df[["S2_current_coverage_pct","S2_context_coverage_pct"]],height=170)
    for product in selected:
        st.caption(f"{ALL_PRODUCTS[product]['label']} · {ALL_PRODUCTS[product]['unit']}")
        st.line_chart(df[[product]],height=180)
    with st.expander("Daily values, coverage and missing observations"):
        st.dataframe(df,use_container_width=True)
    st.download_button("⬇ 30-day observations & coverage CSV",df.to_csv(),file_name="daily_observations_coverage.csv",mime="text/csv",key="month_csv")
    st.caption("Not-yet-requested days and missing observations are NaN, never zero. Use Prepare / resume all 30 days for the full archive. Curves use display-grid means, not an emission inventory.")
errors=[]
for d in days:
    r=load_record(d,config)
    for product,error in r["errors"].items():errors.append({"date":str(d),"product":product,"error":error})
if errors:
    with st.expander(f"Exact request errors ({len(errors)})",expanded=not bool(record.get("s2") or record["products"])):
        error_df=pd.DataFrame(errors);st.dataframe(error_df,use_container_width=True)
        st.download_button("⬇ Error log",error_df.to_csv(index=False),file_name="request_errors.csv",mime="text/csv")
if st.button("📦 Prepare downloaded 30-day archive",disabled=not df.notna().any().any()):
    archive_path=session_folder(config)/"downloaded_archive.zip"
    with st.spinner("Packaging cached rasters…"):
        with zipfile.ZipFile(archive_path,"w",zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("config.json",json.dumps(config,ensure_ascii=False,indent=2))
            archive.writestr("daily_observations_coverage.csv",df.to_csv())
            archive.writestr("request_errors.json",json.dumps(errors,ensure_ascii=False))
            archive.writestr("README.txt","S2 TIFF band order: B02,B03,B04,B08,B11,B12,SCL,dataMask,observation_day. observation_day is integer days since 1970-01-01. S2 is past-only last-known context, not a same-day photograph. S5P bands: daily mean,dataMask,observation_count. Missing retrievals are not zero. All layers are exploratory; no source attribution. config.json defines AOI, QA, grid and screening settings. S2 is a spectral screen, not a validated smoke classifier.")
            for d in days:
                r=load_record(d,config)
                archive.writestr(str(d)+"/status.json",json.dumps(r,ensure_ascii=False,indent=2))
                if r.get("s2"):archive.write(r["s2"]["path"],arcname=str(d)+"/S2_observation_context.tif")
                for product,layer in r["products"].items():
                    archive.write(layer["path"],arcname=str(d)+"/"+product+".tif")
    st.session_state["archive_"+signature]=str(archive_path)
archive_path=st.session_state.get("archive_"+signature)
if archive_path and Path(archive_path).exists():
    st.download_button("⬇ Download prepared archive",Path(archive_path).read_bytes(),file_name="smoke_methane_30days.zip",mime="application/zip",key="archive_download")
