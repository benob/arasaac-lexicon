import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from tqdm import tqdm
import json
import sys

repo_id = "Qwen/Qwen3-VL-8B-Instruct"
model = Qwen3VLForConditionalGeneration.from_pretrained(
    repo_id,
    device_map="auto",
    dtype=torch.bfloat16,
    #attn_implementation="kernels-community/flash-attn2",
).to('cuda')
processor = AutoProcessor.from_pretrained(repo_id)

def generate(image_path, prompt, max_new_tokens=2048):
    messages = [ {
            "role": "user",
            "content": [ 
                { "type": "image", "image": image_path, },
                {"type": "text", "text": prompt},
            ],
    } ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    ).to('cuda')

    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens) #, do_sample=False, cache_implementation="static")
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0]

lexicon = json.load(open('en.json'))

for entry in tqdm(lexicon):
    image_path = f'images/{entry["_id"]}.png'
    keywords = ', '.join([x['keyword'] for x in entry['keywords']])
    prompt = f'Write a detailed description of the image in a single line, knowing that the image is associated with "{keywords}". Do not mention the style of the image. The description must be in English.'
    #print(prompt)
    description = generate(image_path, prompt).split('\n')[0].strip()
    print(entry["_id"], description, sep="\t")
    #print()

