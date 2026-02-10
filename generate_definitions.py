import llm_backend

from tqdm import tqdm
import os
import json

os.environ['VLLM_CONFIGURE_LOGGING'] = '0'

llm = llm_backend.for_name('vllm', 'openai/gpt-oss-120b')

definitions = {}

def load_senses(path):
    with open(path) as fp:
        for line in fp:
            if line.startswith(' ') or '|' not in line:
                continue
            tokens = line.split()
            category = tokens[2]
            synset = tokens[0] + '-' + category
            definition = line.split('|')[-1]
            definitions[synset] = definition

wordnet_path = 'wn-3.1'
for category in ['adj', 'adv', 'noun', 'verb']:
    load_senses(f'{wordnet_path}/data.{category}')

descriptions = {}

import csv
with open('descriptions.csv') as fp:
    images = {}
    reader = csv.reader(fp, delimiter='\t')
    for id, description in reader:
        descriptions[id] = description

lexicon = json.load(open('en.json'))
import random
random.shuffle(lexicon)

for num, entry in enumerate(tqdm(lexicon)):
    id = str(entry['_id'])
    keywords = [x['keyword'] for x in entry['keywords']]
    tags = ', '.join(entry['tags'])
    #categories = ', '.join(entry['categories'])
    synsets = entry['synsets']
    definition = '; '.join([definitions[synset].strip() for synset in synsets if synset in definitions])
    description = descriptions[id]

    prompt = f'''Write a precise, short 10-15 word definition for the word "{keywords[0]}". 
Do not use negations in the definition. 
Do not repeat the lemma unless it is a named entity. 
Only output the definition.
Do not use formatting. 
Indicate the gender of persons if specified in the visual description.
If the word is a named entity, the definition should differenciate it from other entities of the same category.
If the concept is interrogative, evidenced by the presence of a question mark in the visual description, specify it in the definition.
The definition must match the following cues without copying them:
- WordNet definition: {', '.join(keywords)}: {definition}
- Tags: {tags}
- Visual description (in case of ambiguity): {description}
'''
    #print('PROMPT', prompt)
    output = llm.inference(prompt).split('\n')[0].split('\x00')[0].replace('\t', ' ').strip()

    print(id, json.dumps([x['keyword'] for x in entry['keywords']]), output, sep='\t')
