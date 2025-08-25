#!/usr/bin/env python3

import os
import sys
import json
import requests

# Get environment variables
GITHUB_TOKEN = os.environ['GITHUB_TOKEN']
GH_OWNER = os.environ['GH_OWNER']
GH_REPO = os.environ['GH_REPO'] 
TAG_NAME = os.environ['TAG_NAME']
RELEASE_NAME = os.environ['RELEASE_NAME']

# Read release body
with open('release_body.json', 'r') as f:
    release_body = f.read()

# Create release data
release_data = {
    "tag_name": TAG_NAME,
    "name": RELEASE_NAME,
    "body": release_body,
    "draft": False,
    "prerelease": False
}

headers = {
    "Authorization": f"Bearer {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json"
}

# Check if release exists
url = f"https://api.github.com/repos/{GH_OWNER}/{GH_REPO}/releases/tags/{TAG_NAME}"
response = requests.get(url, headers=headers)

if response.status_code == 200:
    # Release exists
    release = response.json()
    print(f"Release already exists: {release['id']}")
else:
    # Create new release
    url = f"https://api.github.com/repos/{GH_OWNER}/{GH_REPO}/releases"
    response = requests.post(url, headers=headers, json=release_data)
    
    if response.status_code == 201:
        release = response.json()
        print(f"Release created: {release['id']}")
    else:
        print(f"Failed to create release: {response.status_code}")
        print(response.text)
        sys.exit(1)

# Output release ID for shell
print(release['id'])