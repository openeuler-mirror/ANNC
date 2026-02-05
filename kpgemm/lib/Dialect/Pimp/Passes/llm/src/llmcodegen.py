import os
import sys
import subprocess
from openai import OpenAI

# clean markdown marks from llm output
def clean_code_block(generated_code):
    lines = generated_code.split('\n')
    
    if lines and lines[0].strip().startswith('```'):
        lines = lines[1:]
    
    if lines and lines[-1].strip() == '```':
        lines = lines[:-1]
    
    cleaned_code = '\n'.join(lines)
    return cleaned_code

client = OpenAI(
    base_url="https://ark.cn-beijing.volces.com/api/v3",
    # api_key=os.environ.get("API_KEY"),
    api_key="c6355d0b-f0b0-41e2-ab72-a65634504163",
)

path = sys.argv[1]
patternName = sys.argv[2]
tfs_prompt = path + "/prompts/tfs_prompt.txt"
generated_files_dir = path + "/outputs/generated_files"
generated_file = os.path.join(generated_files_dir, patternName + ".cc")

os.makedirs(generated_files_dir, exist_ok=True)

with open(tfs_prompt, 'r', encoding='utf-8') as f:
    prompt_content = f.read().strip()
print("Generating Code to: " + generated_file)
completion = client.chat.completions.create(
    model="deepseek-v3-1-terminus",
    messages=[
        {"role": "system", "content": ""},
        {"role": "user", "content": prompt_content},
    ],
)

generated_code = completion.choices[0].message.content
generated_code = clean_code_block(generated_code)

try:
    with open(generated_file, 'w', encoding='utf-8') as f:
        f.write(generated_code)
except Exception as e:
    print(f"\n Failed to Save Generated File: {e}")