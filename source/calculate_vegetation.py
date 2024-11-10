import rasterio
import numpy as np

directory = 'C:\\Users\\Gandhi\\Desktop\\Annaba climate\\'
# Paths to your downloaded B04 and B08 band images
#b04_path = directory+'2023-01-11-00_00_2023-01-11-23_59_Sentinel-2_L2A_B04_(Raw).tiff'
#b08_path = directory+'2023-01-11-00_00_2023-01-11-23_59_Sentinel-2_L2A_B08_(Raw).tiff'
# Open the B04 and B08 band images
'''

b04_path = directory+'2023-02-07-00_00_2023-02-07-23_59_Sentinel-2_L2A_B04_(Raw).tiff'
b08_path = directory+'2023-02-07-00_00_2023-02-07-23_59_Sentinel-2_L2A_B08_(Raw).tiff'

b04_path = directory+'2023-03-17-00_00_2023-03-17-23_59_Sentinel-2_L2A_B04_(Raw).tiff'
b08_path = directory+'2023-03-17-00_00_2023-03-17-23_59_Sentinel-2_L2A_B08_(Raw).tiff'


b04_path = directory+'2023-04-11-00_00_2023-04-11-23_59_Sentinel-2_L2A_B04_(Raw).tiff'
b08_path = directory+'2023-04-11-00_00_2023-04-11-23_59_Sentinel-2_L2A_B08_(Raw).tiff'


b04_path = directory+'2023-05-26-00_00_2023-05-26-23_59_Sentinel-2_L2A_B04_(Raw).tiff'
b08_path = directory+'2023-05-26-00_00_2023-05-26-23_59_Sentinel-2_L2A_B08_(Raw).tiff'

b04_path = directory+'2023-06-12-00_00_2023-06-12-23_59_Sentinel-2_L2A_B04_(Raw).tiff'
b08_path = directory+'2023-06-12-00_00_2023-06-12-23_59_Sentinel-2_L2A_B08_(Raw).tiff'

b04_path = directory+'2023-07-12-00_00_2023-07-12-23_59_Sentinel-2_L2A_B04_(Raw).tiff'
b08_path = directory+'2023-07-12-00_00_2023-07-12-23_59_Sentinel-2_L2A_B08_(Raw).tiff'

b04_path = directory+'2023-08-16-00_00_2023-08-16-23_59_Sentinel-2_L2A_B04_(Raw).tiff'
b08_path = directory+'2023-08-16-00_00_2023-08-16-23_59_Sentinel-2_L2A_B08_(Raw).tiff'

b04_path = directory+'2023-09-15-00_00_2023-09-15-23_59_Sentinel-2_L2A_B04_(Raw).tiff'
b08_path = directory+'2023-09-15-00_00_2023-09-15-23_59_Sentinel-2_L2A_B08_(Raw).tiff'


b04_path = directory+'2023-10-10-00_00_2023-10-10-23_59_Sentinel-2_L2A_B04_(Raw).tiff'
b08_path = directory+'2023-10-10-00_00_2023-10-10-23_59_Sentinel-2_L2A_B08_(Raw).tiff'

b04_path = directory+'2023-11-14-00_00_2023-11-14-23_59_Sentinel-2_L2A_B04_(Raw).tiff'
b08_path = directory+'2023-11-14-00_00_2023-11-14-23_59_Sentinel-2_L2A_B08_(Raw).tiff'
'''
b04_path = directory+'2023-12-12-00_00_2023-12-12-23_59_Sentinel-2_L2A_B04_(Raw).tiff'
b08_path = directory+'2023-12-12-00_00_2023-12-12-23_59_Sentinel-2_L2A_B08_(Raw).tiff'


with rasterio.open(b04_path) as red_src, rasterio.open(b08_path) as nir_src:
    # Read the bands as arrays
    red = red_src.read(1).astype('float64')
    nir = nir_src.read(1).astype('float64')

    # Calculate NDVI
    ndvi = (nir - red) / (nir + red)

    # Handle division by zero
    ndvi[np.isnan(ndvi)] = 0

    # Compute statistics
    ndvi_min = np.min(ndvi)
    ndvi_max = np.max(ndvi)
    ndvi_mean = np.mean(ndvi)

    print(f"NDVI Min: {ndvi_min}")
    print(f"NDVI Max: {ndvi_max}")
    print(f"NDVI Mean: {ndvi_mean}")

