from transformers import AutoTokenizer 
from vllm import LLM, SamplingParams

number_gpus = 1
model_name = "neuralmagic/DeepSeek-R1-Distill-Llama-70B-quantized.w4a16"

tokenizer = AutoTokenizer.from_pretrained(model_name)
sampling_params = SamplingParams(temperature=0.6,max_tokens=256,stop_token_ids=[tokenizer.eos_token_id])
llm = LLM(model=model_name, tensor_parallel_size=number_gpus,trust_remote_code=True)

message_list = [
    [{"role": "user","content":"Who are you? Please respond in pirate speak!"}],

]

prompt_token_ids = [tokenizer.apply_chat_template(messages, add_generation_prompt=True) for messages in message_list]

outputs = llm.generate(prompt_token_ids=prompt_token_ids,sampling_params=sampling_params)

generated_text = [output.outputs[0].text for output in outputs]
print(generated_text)