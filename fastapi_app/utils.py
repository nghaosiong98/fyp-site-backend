from datetime import datetime
import numpy as np
import cv2
import mimetypes
from google.cloud import storage


def byte_to_img(image_string):
    nparr = np.frombuffer(image_string, np.uint8)
    raw_image = cv2.imdecode(nparr, cv2.IMREAD_ANYCOLOR)
    raw_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
    return raw_image


def upload_blob_with_metadata(bucket_name, data_str, source_file_name, metadata):
    destination_blob_name = datetime.now().strftime("%m%d%Y-%H%M%S")
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    content_type = mimetypes.guess_type(source_file_name)
    blob.metadata = metadata
    blob.upload_from_string(data_str, content_type=content_type[0])
    print(
        "File {} uploaded to {}.".format(
            source_file_name, destination_blob_name
        )
    )
