from PIL import Image
import piexif

def convert_to_degrees(dms):
    degrees = dms[0][0] / dms[0][1]
    minutes = dms[1][0] / dms[1][1]
    seconds = dms[2][0] / dms[2][1]

    return degrees + (minutes / 60.0) + (seconds / 3600.0)

def extract_lat_long(image_path):
    try:
        img = Image.open(image_path)

        # Get the EXIF data (binary)
        exif_data = img.info.get("exif")
        if not exif_data:
            print("No EXIF data found in image.")
            return None

        # Load and parse the EXIF data
        exif_dict = piexif.load(exif_data)
        gps_data = exif_dict.get("GPS", {})
        if not gps_data:
            print("No GPS data found in EXIF.")
            return None

       
        lat = gps_data.get(2)       
        lat_ref = gps_data.get(1)   
        lon = gps_data.get(4)       
        lon_ref = gps_data.get(3)   

        if lat and lat_ref and lon and lon_ref:
            latitude = convert_to_degrees(lat)
            if lat_ref == b'S':
                latitude = -latitude

            longitude = convert_to_degrees(lon)
            if lon_ref == b'W':
                longitude = -longitude

            return (latitude, longitude)
        else:
            print("Incomplete GPS info.")
            return None

    except Exception as e:
        print(f"Error reading image metadata: {e}")
        return None
    
# just added this for testing 
if __name__ == "__main__":
    coords = extract_lat_long("/home/vijeth/major-project/backend/PXL_20240921_055604115.jpg")
    if coords:
        print("Latitude:", coords[0])
        print("Longitude:", coords[1])
    else:
        print("No location data found.")
