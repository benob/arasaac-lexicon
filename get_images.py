import json
import sys
import os
from tqdm import tqdm
from PIL import Image
import urllib.request
import io

base_url = 'https://api.arasaac.org/v1/pictograms/'

os.makedirs('images', exist_ok=True)

data = []
for entry in tqdm(json.load(open(sys.argv[1]))):
  id = entry['_id']
  image_path = f"images/{id}.png"
  if not os.path.exists(image_path):
    # add white background and resize to 512x512
    png = Image.open(io.BytesIO(urllib.request.urlopen(base_url + str(id)).read())).convert('RGBA')
    background = Image.new("RGBA", png.size, (255, 255, 255, 255))
    background.paste(png, (0, 0), png)
    background = background.resize((512, 512)).convert('RGB')
    background.save(image_path)
