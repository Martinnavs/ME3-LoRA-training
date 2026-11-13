"""Merge a LoRA adapter into its base model; prints the llama.cpp steps to make the GGUF.

    python train/export.py --base ibm-granite/granite-4.0-h-1b --adapter runs/h1b-lora-s0/final --out runs/h1b-lora-s0/merged

Then (on any machine with llama.cpp; the converter handles the `granitehybrid` architecture, the same family
that produced the GGUF files used in the laptop experiments):
    python llama.cpp/convert_hf_to_gguf.py runs/h1b-lora-s0/merged --outfile router-f16.gguf --outtype f16
    llama.cpp/build/bin/llama-quantize router-f16.gguf router-Q4_K_M.gguf Q4_K_M
Compare Q8_0 as well; see what quantisation costs with eval/onit_eval.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(a.base, torch_dtype=torch.bfloat16)
    merged = PeftModel.from_pretrained(model, str(a.adapter)).merge_and_unload()
    a.out.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(a.out))
    AutoTokenizer.from_pretrained(a.base).save_pretrained(str(a.out))
    print(f"merged model in {a.out}\n" + __doc__.split("Then ", 1)[1])


if __name__ == "__main__":
    main()
