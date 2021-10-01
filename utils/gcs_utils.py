from google.cloud import storage
from zipfile import ZipFile
from zipfile import is_zipfile
import gzip
import io
import os
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = f"{os.path.dirname(__file__)}/../gcloud.json"

gcs_bucket = "capstone-fall21-protein"
storage_client = storage.Client()
bucket = storage_client.get_bucket(gcs_bucket)


def download_text(key):
    blob = bucket.blob(key)
    return blob.download_as_string().decode("utf-8")


def download_gzip(key):
    blob = bucket.blob(key)
    return gzip.open(io.BytesIO(blob.download_as_string()), mode="rt")


def download_gzip_to_string(key):
    with download_gzip(key) as f:
        return f.read()


def uri_to_bucket_and_key(gcs_path):
    return gcs_path.strip("gs://").split("/", 1)


def list_keys(prefix=None):
    return [b.name for b in bucket.list_blobs(prefix=prefix)]


def get_gcs_path(key):
    return f"gs://{gcs_bucket}/{key}"


def list_file_paths(prefix=None):
    keys = list_keys(prefix)
    return [get_gcs_path(k) for k in keys]

def upload_blob(source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    # bucket_name = "your-bucket-name"
    # source_file_name = "local/path/to/file"
    # destination_blob_name = "storage-object-name"

    bucket = storage_client.bucket(gcs_bucket)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

