from cs336_basics import tokenization
import argparse
import json
from pathlib import Path
def train_tokenizer():
    parser = argparse.ArgumentParser(description = "Train a tokenizer on a given text path")
    parser.add_argument("--text_path", type=str, required=True, help="Path to the text file to train the tokenizer on")
    parser.add_argument("--vocab_size", type=int, default=1000, help="Size of the tokenizer vocabulary")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the trained tokenizer")
    args = parser.parse_args()
    from pathlib import Path

    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    # Train the tokenizer   
    SPECIAL_TOKENS = ["<|endoftext|>"]
    vocab, merges = tokenization.train_bpe(args.text_path, args.vocab_size, SPECIAL_TOKENS)
    # Save the tokenizer
    

    special_vocab = {str(token_id): token.hex() for token_id, token in vocab.items()}
    special_merges = [[left.hex(), right.hex()] for left, right in merges]
    
    with open(output_path / "vocab.json", "w", encoding="utf-8") as f:
        json.dump(special_vocab, f)
    with open(output_path / "merges.json", "w", encoding="utf-8") as f:    
        json.dump(special_merges, f)
    with open(output_path / "config.json", "w", encoding="utf-8") as f:
        json.dump({"special_tokens": SPECIAL_TOKENS}, f, indent=2)
    


if __name__ == "__main__":
    train_tokenizer()



        