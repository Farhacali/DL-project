from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

PATH = "nllb-en-ml"
tok = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M",
                                    src_lang="eng_Latn", tgt_lang="mal_Mlym")
model = AutoModelForSeq2SeqLM.from_pretrained(PATH).float().eval()
BOS = tok.convert_tokens_to_ids("mal_Mlym")

app = FastAPI(title="English-Malayalam Translator")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


class Req(BaseModel):
    text: str


@app.post("/translate")
def translate(req: Req):
    x = tok(req.text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        out = model.generate(**x, forced_bos_token_id=BOS,
                             max_length=128, num_beams=4)
    return {"english": req.text,
            "malayalam": tok.decode(out[0], skip_special_tokens=True)}


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>English to Malayalam</title>
<style>
body{font-family:system-ui,"Noto Sans Malayalam",sans-serif;max-width:640px;margin:40px auto;padding:0 16px}
textarea{width:100%;height:100px;font-size:16px;padding:8px;box-sizing:border-box}
button{margin-top:8px;padding:10px 20px;font-size:16px;cursor:pointer}
#out{margin-top:20px;padding:12px;border:1px solid #ccc;border-radius:6px;min-height:48px;font-size:20px}
</style></head><body>
<h2>English to Malayalam</h2>
<textarea id="inp" placeholder="Type an English sentence"></textarea><br>
<button onclick="go()">Translate</button>
<div id="out"></div>
<script>
async function go(){
  const out=document.getElementById('out');
  out.textContent='Translating...';
  try{
    const r=await fetch('/translate',{method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({text:document.getElementById('inp').value})});
    const d=await r.json();
    out.textContent=d.malayalam;
  }catch(e){out.textContent='Error: '+e;}
}
</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
def home():
    return PAGE