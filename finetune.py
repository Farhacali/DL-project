"""Fine-tune NLLB on your CSV.  python finetune.py  [--epochs 3 --batch 8]"""
import argparse
import pandas as pd, torch
from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer,
                          Seq2SeqTrainingArguments, DataCollatorForSeq2Seq)

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/eng_mal_merged.csv")
ap.add_argument("--epochs", type=int, default=5)
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--lr", type=float, default=3e-5)
a = ap.parse_args()

NAME, OUT = "facebook/nllb-200-distilled-600M", "nllb-en-ml"
tok = AutoTokenizer.from_pretrained(NAME, src_lang="eng_Latn", tgt_lang="mal_Mlym")
model = AutoModelForSeq2SeqLM.from_pretrained(NAME)

df = pd.read_csv(a.data).dropna().drop_duplicates()
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
n = len(df); n_test = n_val = int(n * 0.05)
test_df, val_df, train_df = df[:n_test], df[n_test:n_test + n_val], df[n_test + n_val:]
test_df.to_csv("data/test.csv", index=False)   # used by evaluate.py

def prep(b):
    return tok(b["english"], text_target=b["malayalam"], max_length=64, truncation=True)

def make(d):
    return Dataset.from_pandas(d).map(prep, batched=True,
                                      remove_columns=["english", "malayalam"])

args = Seq2SeqTrainingArguments(
    output_dir=OUT,
    per_device_train_batch_size=a.batch,
    per_device_eval_batch_size=a.batch,
    learning_rate=a.lr,
    num_train_epochs=a.epochs,
    weight_decay=0.01,
    eval_strategy="epoch",          # older transformers: evaluation_strategy
    save_strategy="epoch",
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    fp16=torch.cuda.is_available(),
    report_to="none",
)

trainer = Seq2SeqTrainer(model=model, args=args,
                         train_dataset=make(train_df), eval_dataset=make(val_df),
                         data_collator=DataCollatorForSeq2Seq(tok, model=model))
trainer.train()
trainer.save_model(OUT); tok.save_pretrained(OUT)
print("saved to", OUT)