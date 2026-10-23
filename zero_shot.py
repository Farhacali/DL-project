"""Try the pretrained NLLB model with no training.  python zero_shot.py "I am at the market." """
import sys
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

NAME = "facebook/nllb-200-distilled-600M"
tok = AutoTokenizer.from_pretrained(NAME, src_lang="eng_Latn", tgt_lang="mal_Mlym")
model = AutoModelForSeq2SeqLM.from_pretrained(NAME)

text = " ".join(sys.argv[1:]) or "I am at the market."
inputs = tok(text, return_tensors="pt")
out = model.generate(**inputs, forced_bos_token_id=tok.convert_tokens_to_ids("mal_Mlym"),
                     max_length=64, num_beams=4)
print(tok.batch_decode(out, skip_special_tokens=True)[0])