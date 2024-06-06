from vllm import LLM, SamplingParams
import os, json, tempfile

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

llm = LLM(model="camenduru/Meta-Llama-3-8B-Instruct")
tokenizer = llm.get_tokenizer()

def generate(command):
    values = json.loads(command)
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

    return file_path

import gradio as gr

with gr.Blocks(css=".gradio-container {max-width: 544px !important}", analytics_enabled=False) as demo:
    with gr.Row():
        with gr.Column():
            textbox = gr.Textbox(
                show_label=False,
                value="""{
                            "messages": [
                                {
                                "role": "user",
                                "text": "Tell me a story about the Cheesecake Kingdom."
                                }
                            ],
                            "model": {
                                "model": "chat-meta-llama-3-8b-instruct",
                                "max_tokens": 256,
                                "min_tokens": 1,
                                "presence_penalty": 0,
                                "frequency_penalty": 0,
                                "repetition_penalty": 2,
                                "length_penalty": 1,
                                "temperature": 0.8,
                                "top_p": 1,
                                "top_k": 40,
                                "min_p": 0,
                                "ignore_eos": false,
                                "system_prompt": "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."
                            }
                        }"""
            )
            button = gr.Button()
    with gr.Row():
        file = gr.File(
            show_label=False,
            elem_id="file",
        )
    button.click(fn=generate, inputs=[textbox], outputs=[file], show_progress=True)

import os
PORT = int(os.getenv('server_port'))
demo.queue().launch(inline=False, share=False, debug=True, server_name='0.0.0.0', server_port=PORT)