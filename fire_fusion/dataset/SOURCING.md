#### File Loaders
- [xr.open_dataset](https://xarray.pydata.org/en/v2023.08.0/generated/xarray.open_dataset.html)
- [rasterio.open](https://rasterio.readthedocs.io/en/stable/topics/reading.html)
- [rioxarray.open_rasterio](https://corteva.github.io/rioxarray/html/rioxarray.html)
- [geopandas.read_file](https://geopandas.org/en/v1.1.0/docs/reference/api/geopandas.read_file.html)


#### USFS Fire Layers
- [Occurence Point](https://data-usfs.hub.arcgis.com/datasets/usfs%3A%3Anational-usfs-fire-occurrence-point-feature-layer/about?utm_source=chatgpt.com)
- [Perimeter Layer](https://data-usfs.hub.arcgis.com/datasets/usfs::national-usfs-fire-perimeter-feature-layer/about)
- [Docs](https://www.landfire.gov/sites/default/files/documents/LF_Data_Dictionary.pdf)


#### USDA Wildlife-Urban-Interface (WUI)
- [Download](https://www.fs.usda.gov/rds/archive/catalog/RDS-2015-0012-3)
- [Metadata](https://www.fs.usda.gov/rds/archive/products/RDS-2015-0012-4/_metadata_RDS-2015-0012-4.html)
  - **WUIFLAG{year}:**
    - 0: non-WUI
    - 1: intermix
    - 2: interface
  - **WUICLASS{year}:**
    - **Value:** Water
    - Definition: open water
    - **Value:** Med_Dens_Interface
    - Definition: housing density between 49.42108 and 741.3162 and wildland vegetation <= 50% and within 2.414 km of area with >= 75% wildland vegetation
    - **Value:** High_Dens_Intermix
    - Definition: housing density >= 741.3162 and wildland vegetation > 50%
    - **Value:** Low_Dens_Interface
    - Definition: housing density between 6.177635 and 49.42108 and wildland vegetation <= 50% and within 2.414 km of area with >= 75% wildland vegetation
    - **Value:** Med_Dens_Intermix
    - Definition: housing density between 49.42108 and 741.3162 and wildland vegetation > 50%
    - **Value:** Low_Dens_NoVeg
    - Definition: housing density between 6.177635 and 49.42108 and wildland vegetation <= 50%
    - **Value:** Med_Dens_NoVeg
    - Definition: housing density between 49.42108 and 741.3162 and wildland vegetation <= 50%
    - **Value:** Uninhabited_NoVeg
    - Definition: housing density = 0 and wildland vegetation <= 50%
    - **Value:** Low_Dens_Intermix
    - Definition: housing density between 6.177635 and 49.42108 and wildland vegetation > 50%
    - **Value:** High_Dens_Interface
    - Definition: housing density >= 741.3162 and wildland vegetation <= 50% and within 2.414 km of area with >= 75% wildland vegetation
    - **Value:** Very_Low_Dens_Veg
    - Definition: housing density < 6.177635 and wildland vegetation > 50%
    - **Value:** Uninhabited_Veg
    - Definition: housing density = 0 and wildland vegetation > 50%
    - **Value:** Very_Low_Dens_NoVeg
    - Definition:housing density < 6.177635 and wildland vegetation <= 50%
    - **Value:** High_Dens_NoVeg
    - Definition:housing density >= 741.3162 and wildland vegetation <= 50%


#### MODIS:
- [earthaccess API](https://earthaccess.readthedocs.io/en/latest/user-reference/api/api/#earthaccess.api.download)
- [NASA Data Explorer](https://ladsweb.modaps.eosdis.nasa.gov/search/order/1/MYD11A1--61,MCD15A2H--61)
- (MCD152AH)[https://ladsweb.modaps.eosdis.nasa.gov/missions-and-measurements/products/MCD15A2H]
  - (File spec)[https://ladsweb.modaps.eosdis.nasa.gov/filespec/MODIS/6/MCD15A2H]
  - **Lai_500m**
    - 255 fill (bad) value
  - **FparLai_QC** (primary)
    - bit 0: MODLAND (0 = good)
    - bits 3–4: cloud state
    - bits 5–7: SCF_QC (retrieval quality / method)
  - **FparExtra_QC**
    - bits 0–1: land/sea
    - bit 2: snow/ice
    - bit 3: aerosol
    - bit 4: cirrus
    - bit 5: internal cloud
    - bit 6: cloud shadow
- (MOD13Q1)[https://ladsweb.modaps.eosdis.nasa.gov/missions-and-measurements/products/MOD13Q1]
  - (File spec)[https://ladsweb.modaps.eosdis.nasa.gov/filespec/MODIS/6/MOD13Q1]
  - **250m 16 days VI Quality** 
    - bit 0–1: MODLAND_QA
    - bits 2–5: VI usefulness
    - bit 8: adjacent cloud detected
    - bit 10: mixed cloud
    - bits 11–13: land/water flag (1 = land, 0/2/5/6/7=deep water)
    - bits 14–15: possible snow/ice / possible shadow (optional)
  - **250m 16 days NDVI**
    - scale by 1/10,000
- (MCD64A1)[https://ladsweb.modaps.eosdis.nasa.gov/missions-and-measurements/products/MCD64A1]
  - (File spec)[https://ladsweb.modaps.eosdis.nasa.gov/filespec/MODIS/6/MCD64A1]
  - **Burn Date**
    - 1-365 of SINGLE FIRST BURN DAY (0=unburned, 1=burned, -1 unmapped, -2 water)
    - 

**gridMET**: https://www.climatologylab.org/wget-gridmet.html

**USFS Fire**
- https://data-usfs.hub.arcgis.com/datasets/usfs%3A%3Anational-usfs-fire-occurrence-point-feature-layer/about
- https://data-usfs.hub.arcgis.com/datasets/usfs::national-usfs-fire-perimeter-feature-layer/about


#### LANDFIRE
- [Data](https://www.landfire.gov/viewer/)
- [Docs](https://www.landfire.gov/sites/default/files/documents/LF_Data_Dictionary.pdf)
- LANDFIRE 2024. U.S. Geological Survey. https://www.landfire.gov/


#### NLCD
- [Data](https://www.mrlc.gov/viewer/)
- [Docs](https://www.mrlc.gov/sites/default/files/docs/LSDS-2103%20Annual%20National%20Land%20Cover%20Database%20(NLCD)%20Collection%201%20Science%20Product%20User%20Guide%20-v1.1%202025_06_11.pdf)
- Dewitz, J., 2023. National Land Cover Database (NLCD) 2021 Products. https://doi.org/10.5066/P9KZCM54


#### gridMET
- [Data](https://www.climatologylab.org/wget-gridmet.html)
- [Reference](https://planetarycomputer.microsoft.com/api/stac/v1/collections/gridmet)


### TIGER/Line Shapefiles
- [Data](https://www.census.gov/cgi-bin/geo/shapefiles/index.php)


#### GPWv4:
- [Gridded Population of the World](https://search.earthdata.nasa.gov/search?fpj=GPW&oe=t&fsm0=Population&fst0=Human+Dimensions&lat=37.29469630844446&long=-72.07369294814293)
- Gridded Population of the World, Version 4 (GPWv4): Population Density Adjusted to Match 2015 Revision UN WPP Country Totals, Revision 11Version: 4.11Creator: Center for International Earth Science Information Network - CIESIN - Columbia UniversityPublisher: ESDISRelease Date: 2018-12-31T00:00:00.000ZRelease Place: Palisades, NYLinkage: https://doi.org/10.7927/H4F47M65
- https://search.earthdata.nasa.gov/search?q=CIESIN%20ESDIS&hdr=500%2Bto%2B1000%2Bmeters&fpj=GPW&fsm0=Population&fst0=Human%20Dimensions&lat=37.29469630844446&long=-72.07369294814293
- (Earth Search NASA)[https://search.earthdata.nasa.gov/search/granules?p=C3540910651-ESDIS&pg[0][v]=f&pg[0][id]=30_sec&pg[0][gsk]=-start_date&long=31.11328125]

