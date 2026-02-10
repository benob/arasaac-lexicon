# Arasaac lexicon with visual descriptions and word definitions

This repository contains tools to download the Arasaac pictograms for Augmentative and Alternative Communication (AAC). It also contains visual descriptions and definitions for each of the images which have been generated with local LLMs.

## Install dependencies
```
pip install -r requirements.txt
```

## Download metadata from Arasaac
```
curl https://api.arasaac.org/v1/pictograms/all/en -o en.json
```

## Retrieve images from Arasaac website
```
python get_images.py
```

## Generate descriptions with Qwen3-VL-8B-Instruct
```
python generate_descriptions.py > descriptions.csv
```

## Generate definitions with gpt-oss-120b, requires descriptions
```
python generate_definitions.py > definitions.csv
```
