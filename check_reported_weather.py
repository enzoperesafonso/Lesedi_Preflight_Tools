import requests
from datetime import datetime

current_time = datetime.now()

def parse_iso_time(iso_time):
    return datetime.strptime(iso_time, "%Y-%m-%dT%H:%M:%SZ")


def get_predicted_cloud_cover():
    """Access the Norwegian Meteorological Institute Weather to obtain the total current cloud cover percentage over Sutherland Observatory.

    The API used here is the same one used for the Sutherland Observatory Weather Reporting Page hosted at https://suthweather.saao.ac.za .

    :return: Time of request and area of sky obscured by cloud as a percentage value.
    :rtype: String.
    :raises ValueError: Prediction Error.
    """
    api_url = "https://api.met.no/weatherapi/locationforecast/2.0/compact?lat=-32.37998&lon=20.81058"

    # TODO: Ask Dr Erasmus for SAAO email to be associated with meteo api...
    headers = {
        "User-Agent": "https://github.com/enzoperesafonso/LESEDI_CLOUD_CONDITION"
    }
    try:
        response = requests.get(api_url, headers=headers)
        weather_data = response.json()

        closest_entry = min(
            weather_data["properties"]["timeseries"],
            key=lambda x: abs(parse_iso_time(x["time"]) - current_time),
        )

        # Extract cloud cover percentage from the closest entry
        cloud_cover = closest_entry["data"]["instant"]["details"]["cloud_area_fraction"]

        return f"Cloud cover {cloud_cover} % at {current_time}:"
    except Exception as e:
        print("Prediction Error:", e)
