import os
import openai
import json
from openai import OpenAI
from tqdm import tqdm
import time
import random

client=OpenAI(api_key='<KEY>')
client.api_key=''
client.base_url=''

json_name = "./FSC147_prompt_4prompt_30toknes.json"
txt_path = "./data/allImageClasses_FSC147.txt"
class_f = open(txt_path,"r")
classes = class_f.readlines()
classes = [c.strip() for c in classes]
class_f.close()
category_list = classes
all_responses = {}
vowel_list = ['A', 'E', 'I', 'O', 'U']


def try_api(input_prompt, temperature=0, model='gpt-4o'):
    try:
        response = client.chat.completions.create(
                model=model,
                messages=[
                {"role": "user", "content": input_prompt},
                ],
                temperature=temperature,
                max_tokens=30,
				n=5,
				stop="."
                )
        return response
    except Exception as e:
        print(e.message)
        print(e)
        time.sleep(2)
        return try_api(input_prompt, temperature, random.choice(['gpt-4o', 'gpt-4o-mini', 'gpt-4o-2024-11-20']))


for category in tqdm(category_list):
	if category[0] in vowel_list:
		article = "an"
	else:
		article = "a"
	prompts = []

	prompts.append(f"Describe what {article} {category} looks like?")
	prompts.append(f"How can you identify {article} {category}?")
	prompts.append(f"Describe what {article} {category} in a distance look like within 20 words.")
	prompts.append(f"Describe what {article} {category}  in a low resolution photo look like within 20 words.")


	all_result = []
	for curr_prompt in prompts:
		response = try_api(curr_prompt,temperature=0.99)
		for r in range(len(response.choices)):
			result = response.choices[r].message.content
			all_result.append(result.replace("\n\n", "") + ".")

	all_responses[category] = all_result

with open(json_name, 'w') as f:
	json.dump(all_responses, f, indent=4)
