import os
import sys

GCS_STORAGE_KEY = "../gcs.json"

def setup_env_vars():
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = GCS_STORAGE_KEY

