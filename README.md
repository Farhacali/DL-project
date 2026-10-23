# English to Malayalam Translator (API)

Fine-tuned NLLB-200 (600M) model served with FastAPI.

## Setup
```
python -m venv venv
venv\Scripts\activate        # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
```

## Model
Download the model folder here: <PUT-YOUR-DRIVE-OR-HUGGINGFACE-LINK>
Unzip it so that `nllb-en-ml/` sits next to `api.py`, with `model.safetensors`
directly inside it. Size is about 1.2 GB.

## Run
```
uvicorn api:app --port 8000
```
Wait for "Application startup complete" (model loading takes about a minute).
A demo page is at http://127.0.0.1:8000/

## API
POST /translate
Request:  {"text": "I will decide tomorrow."}
Response: {"english": "I will decide tomorrow.", "malayalam": "ഞാൻ നാളെ തീരുമാനിക്കാം."}

## Notes
- The first run needs internet (it downloads the tokenizer from Hugging Face).
- Responses are UTF-8. Use a Malayalam font such as Noto Sans Malayalam.
- Trained on about 4,000 short everyday sentences, so quality drops on long
  or unusual sentences.
- The NLLB model license is CC-BY-NC (non-commercial). Check it before any
  commercial use.
- Change allow_origins=["*"] in api.py to the real site address before deploying.
