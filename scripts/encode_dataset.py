from __future__ import annotations
import argparse

import numpy as np
from pathlib import Path

from cs336_basics import tokenization

def main() -> None:
    parser = argparse.ArgumentParser(description="Encode a text dataset into token IDs and save as .npy files.")
    parser.add_argument("--input_path", type=Path, required=True, help="Path to the input text file.")
    parser.add_argument("--train_output_path", type=Path, required=True, help="Path to save the training token IDs as a .npy file.")
    parser.add_argument("--tokenizer_path", type=Path, required=True, help="Path to the tokenizer JSON file.")

    args = parser.parse_args()

    tokenizer = tokenization.Tokenizer.from_files(
    args.tokenizer_path / "vocab.json",
    args.tokenizer_path / "merges.json",
    args.tokenizer_path / "config.json",
)

    token_ids = []

    with open(args.input_path, "r", encoding="utf-8") as f:
        for token_id in tokenizer.encode_iterable(f):
            token_ids.append(token_id)

    dtype = np.uint16 if len(tokenizer.vocab) <= np.iinfo(np.uint16).max else np.uint32
    token_array = np.asarray(token_ids, dtype=dtype)

    output_path = args.train_output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, token_array)

    print(f"Saved {len(token_array)} tokens to {output_path}")

if __name__ == "__main__":
    main()