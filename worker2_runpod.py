import os, json, tempfile, requests, runpod

from vllm import LLM, SamplingParams

max_model_len = int(os.getenv('max_model_len', default='42496'))

llm = LLM(model="/content/model", max_model_len=max_model_len)
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
    seed = model['seed']
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
                    seed=seed,
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
    try:
        notify_uri = values['notify_uri']
        del values['notify_uri']
        notify_token = values['notify_token']
        del values['notify_token']
        if(notify_uri == "notify_uri"):
            notify_uri = os.getenv('com_camenduru_notify_uri')
            notify_token = os.getenv('com_camenduru_notify_token')
        discord_id = values['discord_id']
        del values['discord_id']
        if(discord_id == "discord_id"):
            discord_id = os.getenv('com_camenduru_discord_id')
        discord_channel = values['discord_channel']
        del values['discord_channel']
        if(discord_channel == "discord_channel"):
            discord_channel = os.getenv('com_camenduru_discord_channel')
        discord_token = values['discord_token']
        del values['discord_token']
        if(discord_token == "discord_token"):
            discord_token = os.getenv('com_camenduru_discord_token')
        job_id = values['job_id']
        del values['job_id']
        default_filename = os.path.basename(result)
        with open(result, "rb") as file:
            files = {default_filename: file.read()}
        payload = {"content": f"{json.dumps(values)} <@{discord_id}>"}
        response = requests.post(
            f"https://discord.com/api/v9/channels/{discord_channel}/messages",
            data=payload,
            headers={"Authorization": f"Bot {discord_token}"},
            files=files
        )
        response.raise_for_status()
        result_url = response.json()['attachments'][0]['url']
        notify_payload = {"jobId": job_id, "result": result_url, "status": "DONE"}
        requests.post(notify_uri, data=json.dumps(notify_payload), headers={'Content-Type': 'application/json', "Authorization": notify_token})
        return {"jobId": job_id, "result": result_url, "status": "DONE"}
    except Exception as e:
        error_payload = {"jobId": job_id, "status": "FAILED"}
        try:
            requests.post(notify_uri, data=json.dumps(error_payload), headers={'Content-Type': 'application/json', "Authorization": notify_token})
        except:
            pass
        return {"jobId": job_id, "result": f"FAILED: {str(e)}", "status": "FAILED"}
    finally:
        if os.path.exists(result):
            os.remove(result)

runpod.serverless.start({"handler": generate})