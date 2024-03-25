import cv_analysis
import scrape_atlas_image
import datetime

try:
    current_date_and_time = datetime.datetime.now()
    cv_analysis.analyse_cloud_cover(scrape_atlas_image.download_allsky_image(),current_date_and_time)
except:
    print("unable to check all sky image...")