"""Use the fine-tuned model.  python translate.py  |  python translate.py "text" """
import sys, torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

PATH = "nllb-en-ml"
tok = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M",
                                    src_lang="eng_Latn", tgt_lang="mal_Mlym")
model = AutoModelForSeq2SeqLM.from_pretrained(PATH).eval()

def translate(text):
    x = tok(text, return_tensors="pt")
    with torch.no_grad():
        g = model.generate(**x, forced_bos_token_id=tok.convert_tokens_to_ids("mal_Mlym"),
                           max_length=64, num_beams=4)
    return tok.decode(g[0], skip_special_tokens=True)

if len(sys.argv) > 1:
    print(translate(" ".join(sys.argv[1:])))
else:
    while (s := input("English (blank to quit): ").strip()):
        print("Malayalam:", translate(s))