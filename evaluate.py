"""Compare base vs fine-tuned model on the held-out test split."""
import pandas as pd, torch, sacrebleu
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def run(path, sources, bs=16):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M",
                                    src_lang="eng_Latn", tgt_lang="mal_Mlym")
    model = AutoModelForSeq2SeqLM.from_pretrained(path).to(dev).eval()
    outs = []
    for i in range(0, len(sources), bs):
        batch = tok(sources[i:i + bs], return_tensors="pt", padding=True).to(dev)
        with torch.no_grad():
            g = model.generate(**batch, forced_bos_token_id=tok.convert_tokens_to_ids("mal_Mlym"),
                               max_length=64, num_beams=4)
        outs += tok.batch_decode(g, skip_special_tokens=True)
    return outs

test = pd.read_csv("data/test.csv")
src, ref = test["english"].tolist(), test["malayalam"].tolist()
for path in ["facebook/nllb-200-distilled-600M", "nllb-en-ml"]:
    hyp = run(path, src)
    print(f"{path}\n  BLEU {sacrebleu.corpus_bleu(hyp, [ref]).score:.2f} | "
          f"chrF {sacrebleu.corpus_chrf(hyp, [ref]).score:.2f}")
    for s, r, h in list(zip(src, ref, hyp))[:5]:
        print(f"  EN : {s}\n  REF: {r}\n  HYP: {h}\n")