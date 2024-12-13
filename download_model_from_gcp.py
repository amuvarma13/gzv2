import os
from google.cloud import storage
from tqdm import tqdm

def download_folder(bucket_name, output_dir='./mymodel'):
    client = storage.Client.from_service_account_json('service.json')
    bucket = client.bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix='models/checkpoint-5000'))
    for blob in tqdm(blobs, desc="Downloading files"):
        local_path = os.path.join(output_dir, os.path.relpath(blob.name, 'models/checkpoint-5000'))
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        tqdm.write(f"Downloading {blob.name}")
        blob.download_to_filename(local_path)

# Example call:
download_folder('snac-tttts-2m-5000-1')

