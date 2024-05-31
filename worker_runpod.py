import os, json, tempfile, requests, runpod

from vllm import LLM, SamplingParams

discord_token = os.getenv('com_camenduru_discord_token')
web_uri = os.getenv('com_camenduru_web_uri')
web_token = os.getenv('com_camenduru_web_token')

llm = LLM(model="camenduru/Meta-Llama-3-8B-Instruct")
tokenizer = llm.get_tokenizer()

def generate(input):
    values = input["input"]
    messages = values['messages']
    model = values['model']
    prompt = messages[0]['text']
    model_name = model['model']

    max_tokens = model['max_tokens']
    min_tokens = model['min_tokens']
    presence_penalty = model['presence_penalty']
    frequency_penalty = model['frequency_penalty']
    repetition_penalty = model['repetition_penalty']
    length_penalty = model['length_penalty']
    temperature = model['temperature']
    top_p = model['top_p']
    top_k = model['top_k']
    min_p = model['min_p']
    ignore_eos = model['ignore_eos']

    system_prompt = model['system_prompt']
    # template = f"{system_prompt} {prompt}"

    sampling_params = SamplingParams(
                    max_tokens=max_tokens,
                    min_tokens=min_tokens,
                    presence_penalty=presence_penalty,
                    frequency_penalty=frequency_penalty,
                    repetition_penalty=repetition_penalty,
                    length_penalty=length_penalty,
                    temperature=temperature, 
                    top_p=top_p, 
                    top_k=top_k,
                    min_p=min_p,
                    ignore_eos=ignore_eos,
                    stop_token_ids=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")],
                    )
    
    message = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    prompt_with_system = tokenizer.apply_chat_template(message, add_generation_prompt=True, tokenize=False)

    outputs = llm.generate(prompt_with_system, sampling_params)
    text_to_save = outputs[0].outputs[0].text

    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as temp_file:
        file_path = temp_file.name
        temp_file.write(text_to_save.encode())

    result = file_path

    response = None
    try:
        source_id = values['source_id']
        del values['source_id']
        source_channel = values['source_channel']     
        del values['source_channel']
        job_id = values['job_id']
        del values['job_id']
        files = {f"video.mp4": open(result, "rb").read()}
        payload = {"content": f"{json.dumps(values)} <@{source_id}>"}
        response = requests.post(
            f"https://discord.com/api/v9/channels/{source_channel}/messages",
            data=payload,
            headers={"authorization": f"Bot {discord_token}"},
            files=files
        )
        response.raise_for_status()
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        if os.path.exists(result):
            os.remove(result)

    if response and response.status_code == 200:
        try:
            payload = {"jobId": job_id, "result": response.json()['attachments'][0]['url']}
            requests.post(f"{web_uri}/api/notify", data=json.dumps(payload), headers={'Content-Type': 'application/json', "authorization": f"{web_token}"})
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
        finally:
            return {"result": response.json()['attachments'][0]['url']}
    else:
        return {"result": "ERROR"}

runpod.serverless.start({"handler": generate})