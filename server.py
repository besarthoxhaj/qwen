#
#
#
import torch
import asyncio
import fastapi
import pathlib
import pydantic
from fastapi.responses import HTMLResponse, StreamingResponse
import transformers
import uvicorn


#
#
#
tkzr = transformers.AutoTokenizer.from_pretrained('Qwen/Qwen2-0.5B-Instruct')
qwen = transformers.AutoModelForCausalLM.from_pretrained('Qwen/Qwen2-0.5B-Instruct')
trms = {tkzr.eos_token_id, tkzr.convert_tokens_to_ids('<|endoftext|>')}
temp = 0.7


#
#
#
class ChatReq(pydantic.BaseModel):
  history: dict


#
#
#
app = fastapi.FastAPI()


#
#
#
@app.get('/')
def read_root():
  html_content = pathlib.Path('./app.html').read_text()
  return HTMLResponse(content=html_content)


#
#
#
def generate_stream(history):

  conv = [{ 'role': 'system', 'content': 'You are a helpful assistant.' }]
  if history['file'] is not None: conv.append({ 'role': 'user', 'content': f'File: {history["file"]}' })
  conv.extend(history['msgs'])

  txt = tkzr.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
  acc = tkzr(txt, return_tensors='pt').input_ids.to(qwen.device)
  with torch.no_grad(): out = qwen(acc, use_cache=True)
  kvc = out.past_key_values
  lgt = out.logits[:, -1, :] / temp

  for _ in range(256):
    prb = torch.softmax(lgt, dim=-1)
    nex = torch.multinomial(prb, 1)
    tok = nex.item()
    if tok in trms: break
    yield tkzr.decode(tok, skip_special_tokens=True)
    with torch.no_grad(): out = qwen(nex, past_key_values=kvc, use_cache=True)
    kvc = out.past_key_values
    lgt = out.logits[:, -1, :] / temp


#
#
#
@app.post('/chat')
async def chat(req: ChatReq):
  async def token_stream():
    for chunk in generate_stream(req.history):
      yield chunk
      await asyncio.sleep(0)
  return StreamingResponse(token_stream(), media_type='text/plain')


#
#
#
if __name__ == '__main__':
  uvicorn.run(app, host='0.0.0.0', port=8080)
