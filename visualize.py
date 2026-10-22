import rasterio
import matplotlib.pyplot as plt
import numpy as np
import argparse
from pathlib import Path

def tif_to_png(tif_path):
    tif_path = Path(tif_path)
    if not tif_path.exists():
        print(f"Error: {tif_path} does not exist.")
        return

    print(f"Reading {tif_path}...")
    with rasterio.open(tif_path) as src:
        data = src.read(1)
        # Handle empty/transparent pixels
        nodata = src.nodata
        if nodata is not None:
            data = np.ma.masked_equal(data, nodata)
        data = np.ma.masked_invalid(data)

    # Create the plot
    plt.figure(figsize=(12, 10))
    
    # 'turbo' is a beautiful thermal colormap going from Blue (Cold) to Red (Hot)
    im = plt.imshow(data, cmap='turbo')
    
    # Add a color scale bar
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_label('Predicted Temperature (°C)', fontsize=14, weight='bold')
    cbar.ax.tick_params(labelsize=12)
    
    plt.title(f"Urban Heat Island Map\n{tif_path.stem}", fontsize=16, weight='bold', pad=20)
    plt.axis('off') # Hide axes

    # Save as high quality PNG
    png_path = tif_path.with_suffix('.png')
    plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Successfully saved colored heat map to: {png_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a temperature .tif to a colored .png")
    parser.add_argument("tif_file", help="Path to the .tif file")
    args = parser.parse_args()
    
    tif_to_png(args.tif_file)
