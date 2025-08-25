#!/usr/bin/env python3

import os
import sys
import requests
import urllib.parse

# Get environment variables
GITHUB_TOKEN = os.environ['GITHUB_TOKEN']
GH_OWNER = os.environ['GH_OWNER']
GH_REPO = os.environ['GH_REPO']
RELEASE_ID = os.environ['RELEASE_ID']
WHEEL_PATH = os.environ['WHEEL_PATH']
WHEEL_FILE = os.environ['WHEEL_FILE']

headers = {
    "Authorization": f"Bearer {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json"
}

# Check for existing asset and delete if present
assets_url = f"https://api.github.com/repos/{GH_OWNER}/{GH_REPO}/releases/{RELEASE_ID}/assets"
response = requests.get(assets_url, headers=headers)

if response.status_code == 200:
    assets = response.json()
    for asset in assets:
        if asset['name'] == WHEEL_FILE:
            print(f"Deleting existing asset: {asset['id']}")
            delete_url = f"https://api.github.com/repos/{GH_OWNER}/{GH_REPO}/releases/assets/{asset['id']}"
            requests.delete(delete_url, headers=headers)
            break

# Upload new asset
print(f"Uploading {WHEEL_FILE} ({os.path.getsize(WHEEL_PATH) / (1024*1024):.1f} MB)...")

upload_headers = {
    "Authorization": f"Bearer {GITHUB_TOKEN}",
    "Content-Type": "application/octet-stream"
}

encoded_name = urllib.parse.quote(WHEEL_FILE)
upload_url = f"https://uploads.github.com/repos/{GH_OWNER}/{GH_REPO}/releases/{RELEASE_ID}/assets?name={encoded_name}"

with open(WHEEL_PATH, 'rb') as f:
    response = requests.post(upload_url, headers=upload_headers, data=f)

if response.status_code == 201:
    asset = response.json()
    print(f"✓ Asset uploaded successfully!")
    print(f"  Asset ID: {asset['id']}")
    print(f"  Download URL: {asset['browser_download_url']}")
    
    # Write asset ID to file for later use
    with open('/tmp/lrtt_asset_id.txt', 'w') as f:
        f.write(f"ASSET_ID={asset['id']}\n")
    
    print(asset['id'])  # For shell capture
else:
    print(f"✗ Failed to upload asset: {response.status_code}")
    print(response.text)
    sys.exit(1)