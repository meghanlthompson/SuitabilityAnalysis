#GEOG 5092 Final Project
#Title: A Suitability Analysis to Identify Ideal Areas for Asphalt Quarries
#Author: Meghan Thompson
#December 2023

import os
import numpy as np
import geopandas as gpd
import rasterio
from rasterio import features
from rasterio.plot import show
from scipy.ndimage import distance_transform_edt
from rasterio.warp import reproject, Resampling

#######################################################################
# STEP 1: Read the shapefiles necessary for the analysis
Geology_shp = '/Users/meghanthompson/GEOG5092/Project/CO'
Geology = gpd.read_file(Geology_shp)

Railroad_shp = '/Users/meghanthompson/GEOG5092/Project/North_American_Rail_Network_Lines/north_american_rail_network_lines__narn_lines_2023_1.shp'
Railroad = gpd.read_file(Railroad_shp)

Landowner_shp = '/Users/meghanthompson/GEOG5092/Project/GilpinLandOwner'
Landowner = gpd.read_file(Landowner_shp)

Bound_shp = '/Users/meghanthompson/GEOG5092/Project/GilpinBoundary'
Bound = gpd.read_file(Bound_shp)

########################################################################
#STEP 2: Reproject each GeoDataFrame to NAD 83 UTM Zone 13N
Geology_reprojected = Geology.to_crs(epsg=26913)
Railroad_reprojected = Railroad.to_crs(epsg=26913)
Landowner_reprojected = Landowner.to_crs(epsg=26913)
Bound_reprojected = Bound.to_crs(epsg=26913)

# Directory to save the reprojected shapefiles
output_directory = '/Users/meghanthompson/GEOG5092/Project'

# Save the reprojected shapefiles to the specified directory
Geology_reprojected.to_file(os.path.join(output_directory, 'Geology_reprojected.shp'))
Railroad_reprojected.to_file(os.path.join(output_directory, 'Railroad_reprojected.shp'))
Landowner_reprojected.to_file(os.path.join(output_directory, 'Landowner_reprojected.shp'))
Bound_reprojected.to_file(os.path.join(output_directory, 'Bound_reprojected.shp'))

# Paths to reprojected GeoDataFrames
reprojected_files = {
    'Geology': '/Users/meghanthompson/GEOG5092/Project/Geology_reprojected.shp',
    'Railroad': '/Users/meghanthompson/GEOG5092/Project/Railroad_reprojected.shp',
    'Landowner': '/Users/meghanthompson/GEOG5092/Project/Landowner_reprojected.shp',
    'Bound_reprojected': '/Users/meghanthompson/GEOG5092/Project/Bound_reprojected.shp'
}

# Read reprojected shapefiles into GeoDataFrames
Geology_reprojected = gpd.read_file(reprojected_files['Geology'])
Railroad_reprojected = gpd.read_file(reprojected_files['Railroad'])
Landowner_reprojected = gpd.read_file(reprojected_files['Landowner'])
Bound_reprojected = gpd.read_file(reprojected_files['Bound_reprojected'])

#############################################################################
# STEP 3: Clip each shapefile to the bound shapefile and save as new shapefiles
output_directory = '/Users/meghanthompson/GEOG5092/Project'

# List of reprojected GeoDataFrames to be clipped
reprojected_shapefiles = [Geology_reprojected, Railroad_reprojected, Landowner_reprojected]
shapefile_names = ['Geology', 'Railroad', 'Landowner']  # Names for reference

# List to store file paths of clipped shapefiles
clipped_shapefiles_paths = []

# Perform clipping operation and save as new shapefiles
for i, shp in enumerate(reprojected_shapefiles):
    clipped = gpd.clip(shp, Bound_reprojected)
    output_filename = f'{shapefile_names[i]}_clipped.shp'  # New filename for each clipped shapefile
    output_path = os.path.join(output_directory, output_filename)
    clipped.to_file(output_path)
    clipped_shapefiles_paths.append(output_path)

# Display the paths of the saved clipped shapefiles
print("Paths of the saved clipped shapefiles:")
for path in clipped_shapefiles_paths:
    print(path)

# Directory where clipped shapefiles are stored
clipped_directory = '/Users/meghanthompson/GEOG5092/Project'

# List all files in the directory
all_clipped_files = os.listdir(clipped_directory)

# Filter only the shapefiles ending with '_clipped.shp'
clipped_shapefiles = [filename for filename in all_clipped_files if filename.endswith('_clipped.shp')]

# Dictionary to store GeoDataFrames for each clipped shapefile
clipped_dataframes = {}

# Read each clipped shapefile into a GeoDataFrame and store in the dictionary
for shapefile in clipped_shapefiles:
    file_path = os.path.join(clipped_directory, shapefile)
    # Remove the file extension to use as a key in the dictionary
    key = shapefile.replace('.shp', '')
    clipped_dataframes[key] = gpd.read_file(file_path)

#############################################################################
#STEP 4: Rastarize Railroad geodataframe and perform Euclidean Distance
Railroad_clipped = clipped_dataframes['Railroad_clipped']

# Use Railroad_clipped for bounding box calculations
bounds = Railroad_clipped.total_bounds

# Define raster parameters
output_resolution = 10  # Resolution of the output raster in meters
output_crs = 'EPSG:26913'  # CRS of the output raster

# Define the output file path for the Euclidean distance raster
output_path_euclidean = '/Users/meghanthompson/GEOG5092/Project/railroad_euclidean_distance.tif'

# Prepare the raster metadata based on the bounds of the clipped railroad
bounds = Railroad_clipped.total_bounds
width = int((bounds[2] - bounds[0]) / output_resolution)
height = int((bounds[3] - bounds[1]) / output_resolution)
transform = rasterio.transform.from_origin(bounds[0], bounds[3], output_resolution, output_resolution)

# Create a new raster
with rasterio.open(output_path_euclidean, 'w', driver='GTiff', width=width, height=height,
                   count=1, dtype=rasterio.float64, crs=output_crs, transform=transform) as dst:
    # Rasterize the railroad into the new raster
    shapes = ((geom, 1) for geom in Railroad_clipped.geometry)
    rasterized_railroad = features.rasterize(shapes=shapes, out_shape=(height, width), transform=transform,
                                             fill=0, default_value=1, dtype=rasterio.uint8)

    # Perform Euclidean Distance Transform on the rasterized railroad
    euclidean_distance = distance_transform_edt(rasterized_railroad == 0, sampling=[output_resolution, output_resolution])

    # Write the Euclidean distance data to the raster
    dst.write(euclidean_distance, 1)

print("Euclidean distance raster for railroad created and saved.")

# Path to the TIF file (Euclidean distance raster)
tif_path = '/Users/meghanthompson/GEOG5092/Project/railroad_euclidean_distance.tif'

# Open the TIF file
with rasterio.open(tif_path) as src:
    # Read the raster data
    raster_data = src.read(1)  # Assuming it's a single-band raster

    # Visualize the raster data using matplotlib
    show(raster_data, cmap='viridis')
################################################################################
#STEP 5: Create boolean layer for Railroads with the criteria that areas must be within 1 mile of tracks
# Path to store the boolean layer
Railroad_boolean_path = '/Users/meghanthompson/GEOG5092/Project/railroad1mile.tif'

# Open the Euclidean distance raster
with rasterio.open(tif_path) as src:
    # Read the raster data
    raster_data = src.read(1)  # Assuming it's a single-band raster

    # Threshold to identify areas within 1 mile of the railroad
    threshold_distance = 1610  # Distance in meters
    within_1610m = np.where(raster_data <= threshold_distance, 1, 0).astype(np.uint8)

    # Update the metadata for the boolean layer
    profile = src.profile
    profile.update(dtype=rasterio.uint8, count=1)

    # Write the boolean layer to a new raster file
    with rasterio.open(Railroad_boolean_path, 'w', **profile) as dst:
        dst.write(within_1610m, 1)

print("Boolean layer identifying areas within 1 mile of the railroad created and saved.")

# Open the boolean TIFF file
with rasterio.open(Railroad_boolean_path) as src:
    # Read the raster data
    boolean_raster = src.read(1)  # Assuming it's a single-band raster

    # Visualize the boolean raster using matplotlib
    show(boolean_raster, cmap='Greys', title='Areas within 1 Mile of Railroad')

########################################################################################
#STEP 6: Rasterize geology geodataframe and create boolean layer with the criteria that geology type = Xg
# Assuming 'Geology_clipped' is the GeoDataFrame from 'clipped_dataframes'
geology_clipped = clipped_dataframes['Geology_clipped']

# Define the specific criteria based on attributes
specific_criteria = 'Xg'

# Path to store the boolean raster for the specific criteria
geology_boolean_path = '/Users/meghanthompson/GEOG5092/Project/geology_Xg_boolean.tif'

# Define raster parameters
output_resolution = 10  # Resolution of the output raster in meters
output_crs = 'EPSG:26913'  # CRS of the output raster

# Get the bounds for the clipping boundary (Bound_reprojected)
bounds = Bound_reprojected.total_bounds
width = int((bounds[2] - bounds[0]) / output_resolution)
height = int((bounds[3] - bounds[1]) / output_resolution)
transform = rasterio.transform.from_origin(bounds[0], bounds[3], output_resolution, output_resolution)

# Create a new raster
with rasterio.open(geology_boolean_path, 'w', driver='GTiff', width=width, height=height,
                   count=1, dtype=rasterio.uint8, crs=output_crs, transform=transform) as dst:
    # Filter the GeoDataFrame based on specific criteria
    filtered_geology = geology_clipped[geology_clipped['ORIG_LABEL'] == specific_criteria]

    # Rasterize the specific criteria into the new raster
    shapes = ((geom, 1) for geom in filtered_geology.geometry)
    rasterized_geology = features.rasterize(shapes=shapes, out_shape=(height, width), transform=transform,
                                            fill=0, default_value=1, dtype=rasterio.uint8)

    # Write the rasterized data to the raster
    dst.write(rasterized_geology, 1)

print("Boolean raster for specific geology criteria created and saved.")

# Path to the geology boolean raster
geology_boolean_path = '/Users/meghanthompson/GEOG5092/Project/geology_Xg_boolean.tif'

# Open the geology boolean raster
with rasterio.open(geology_boolean_path) as src:
    # Read the raster data
    geology_raster = src.read(1)  # Assuming it's a single-band raster

    # Visualize the boolean raster using matplotlib
    show(geology_raster, cmap='Greys', title='Geology Boolean (Xg)')

###########################################################################################
#STEP 7: Rasterize landowner geodataframe and create boolean layer with the criteria that landowner = private
# Assuming 'Landowners_clipped' is the GeoDataFrame from 'clipped_dataframes'
landowners_clipped = clipped_dataframes['Landowner_clipped']

# Define the specific criteria based on attributes
specific_criteria = 'Private'

# Path to store the boolean raster for the specific criteria
landowners_boolean_path = '/Users/meghanthompson/GEOG5092/Project/landowners_private_boolean.tif'

# Define raster parameters
output_resolution = 10  # Resolution of the output raster in meters
output_crs = 'EPSG:26913'  # CRS of the output raster

# Get the bounds for the clipping boundary (Bound_reprojected)
bounds = Bound_reprojected.total_bounds
width = int((bounds[2] - bounds[0]) / output_resolution)
height = int((bounds[3] - bounds[1]) / output_resolution)
transform = rasterio.transform.from_origin(bounds[0], bounds[3], output_resolution, output_resolution)

# Create a new raster
with rasterio.open(landowners_boolean_path, 'w', driver='GTiff', width=width, height=height,
                   count=1, dtype=rasterio.uint8, crs=output_crs, transform=transform) as dst:
    # Filter the GeoDataFrame based on specific criteria
    filtered_landowners = landowners_clipped[landowners_clipped['OWNER_DETA'] == specific_criteria]

    # Rasterize the specific criteria into the new raster
    shapes = ((geom, 1) for geom in filtered_landowners.geometry)
    rasterized_landowners = features.rasterize(shapes=shapes, out_shape=(height, width), transform=transform,
                                               fill=0, default_value=1, dtype=rasterio.uint8)

    # Write the rasterized data to the raster
    dst.write(rasterized_landowners, 1)

print("Boolean raster for specific landowners criteria created and saved.")

# Path to the landowners boolean raster
landowners_boolean_path = '/Users/meghanthompson/GEOG5092/Project/landowners_private_boolean.tif'

# Open the landowners boolean raster
with rasterio.open(landowners_boolean_path) as src:
    # Read the raster data
    landowners_raster = src.read(1)  # Assuming it's a single-band raster

    # Visualize the boolean raster using matplotlib
    show(landowners_raster, cmap='Greys', title='Landowners Boolean (Private)')

##################################################################################
#STEP 8: Overlay all 3 boolean rasters to find areas that contain all 3 criteria
# Open the boolean rasters
with rasterio.open(Railroad_boolean_path) as src1, \
        rasterio.open(geology_boolean_path) as src2, \
        rasterio.open(landowners_boolean_path) as src3:
    # Read the raster data and get the metadata
    railroad_data = src1.read(1)
    geology_data = src2.read(1)
    landowners_data = src3.read(1)
    profile = src1.profile  # Use the profile from any of the rasters

    # Reproject and resample the rasters to have the same shape
    # Resample to the resolution and extent of the railroad raster (example)
    resampled_geology = np.empty_like(railroad_data)
    reproject(
        geology_data, resampled_geology,
        src_transform=src2.transform, src_crs=src2.crs,
        dst_transform=src1.transform, dst_crs=src1.crs,
        resampling=Resampling.nearest
    )

    resampled_landowners = np.empty_like(railroad_data)
    reproject(
        landowners_data, resampled_landowners,
        src_transform=src3.transform, src_crs=src3.crs,
        dst_transform=src1.transform, dst_crs=src1.crs,
        resampling=Resampling.nearest
    )

    # Perform the logical "AND" operation
    combined_raster = (railroad_data == 1) & (resampled_geology == 1) & (resampled_landowners == 1)

    # Set areas meeting all criteria to 1, others to 0
    combined_raster = combined_raster.astype(np.uint8)

    # Write the combined raster to a new raster file
    combined_raster_path = '/Users/meghanthompson/GEOG5092/Project/combined_criteria_boolean.tif'
    profile.update(dtype=rasterio.uint8, count=1)  # Update dtype and count
    with rasterio.open(combined_raster_path, 'w', **profile) as dst:
        dst.write(combined_raster, 1)

print
#################################################################################################
#STEP 9: Calculate area of suitable land
# Path to the combined boolean raster
combined_raster_path = '/Users/meghanthompson/GEOG5092/Project/combined_criteria_boolean.tif'

# Open the combined boolean raster
with rasterio.open(combined_raster_path) as src:
    # Read the raster data
    combined_raster = src.read(1)

    # Visualize the combined boolean raster using matplotlib
    show(combined_raster, cmap='Greys', title='Combined Boolean Raster')

    # Get the metadata
    transform = src.transform  # Affine transformation matrix
    count_value = 1  # Value representing areas meeting all criteria

    # Count the number of pixels with the desired value
    count = (combined_raster == count_value).astype('uint64').sum()  # Cast to larger data type

    # Calculate the area in square meters
    pixel_area = abs(transform.a * transform.e)  # Pixel area in square meters
    total_area_meters = count * pixel_area  # Total area in square meters

    # Convert square meters to square miles and square hectares
    square_miles = total_area_meters * 3.861e-7
    square_hectares = total_area_meters * 0.0001

print(f"Total area of areas meeting all criteria:")
print(f"In square meters: {total_area_meters:.2f} square meters")
print(f"In square miles: {square_miles:.2f} square miles")
print(f"In square hectares: {square_hectares:.2f} square hectares")