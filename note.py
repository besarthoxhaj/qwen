#%%
import torch
import transformers
#%%
tkzr = transformers.AutoTokenizer.from_pretrained('Qwen/Qwen2-0.5B-Instruct')
qwen = transformers.AutoModelForCausalLM.from_pretrained('Qwen/Qwen2-0.5B-Instruct')
trms = {tkzr.eos_token_id, tkzr.convert_tokens_to_ids('<|endoftext|>')}
temp = 0.7
#%%
sys = {'role': 'system', 'content': 'You are a helpful assistant.'}
usr = {'role': 'user', 'content': 'Give me a 2 sentence introduction to large language models.'}
msg = [sys, usr]
txt = tkzr.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
acc = tkzr(txt, return_tensors='pt').input_ids.to(qwen.device)

for _ in range(256):
  with torch.no_grad():
    lgt = qwen(acc).logits[:, -1, :] / temp
    prb = torch.softmax(lgt, dim=-1)
    nex = torch.multinomial(prb, 1)
  acc = torch.cat([acc, nex], dim=-1)
  tok_id = nex.item()
  if tok_id in trms: break
  res = tkzr.decode(tok_id, skip_special_tokens=True)
  print(res)