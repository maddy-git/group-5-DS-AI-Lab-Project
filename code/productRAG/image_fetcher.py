import os
from . import constants

def get_image_path(sku):
    try:
        base_dir = constants.images_dir
        sku_str = "0" + str(sku).strip()
        if len(sku_str) < 3:
            raise ValueError(f"Invalid SKU: {sku}")

        subfolder = sku_str[:3]  # first 3 digits
        return os.path.join(base_dir, subfolder, f"{sku_str}.jpg")
    except:
        return None