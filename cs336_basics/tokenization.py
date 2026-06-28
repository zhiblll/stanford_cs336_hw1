# import regex as re
import regex as re
from collections import Counter, defaultdict
import json

def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]):
    vocab = {}
    merges = []
    for i in range(256):
        vocab[i] = bytes([i])
    
    cur_pos = 256
    for token in special_tokens:
        vocab[cur_pos] = token.encode(encoding ="utf-8")
        cur_pos = cur_pos +1
    
    # pre_words_freq = Counter()
    
    PRETOKEN_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    pre_words_freq = Counter()

    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    
    
    
    seperate_senten = seperate_sentence(special_tokens, text, keep_special_token=False)
    
    
    for seg in seperate_senten:
        for match in re.finditer(PRETOKEN_PATTERN, seg):
            pretoken = match.group(0)
            bs = pretoken.encode(encoding ="utf-8")
            token_seq = tuple(bytes([b]) for b in bs)
            pre_words_freq[token_seq] += 1
    
    pre_count, pair_to_word = count_freq(pre_words_freq)
    
    while len(vocab) < vocab_size:
        
        if not pre_count:
            break
    
        best_pair = find_best_pair(pre_count)

        new_token = best_pair[0] + best_pair[1]
        merges.append(best_pair)
        vocab[cur_pos] = new_token
        cur_pos += 1
        # words = pair_to_word[best_pair]
        
        merged_words = []
        
        for word in pair_to_word[best_pair].copy():
            freq = pre_words_freq[word]
            
            new_word = merge_pair(best_pair, word)
            remoev_word_contribution(pre_count, word, freq, pair_to_word)
            del pre_words_freq[word]
            pre_words_freq[new_word] += freq
            add_word_contribution(pre_count, new_word, freq, pair_to_word)
            merged_words.append(new_word)
    
    return vocab, merges 
        
     
            
def seperate_sentence(special_tokens, text, keep_special_token):
    if not special_tokens:
        return [text] if text else []
    
    
    escaped = [re.escape(token) for token in sorted(special_tokens, key=len, reverse=True)]
    pattern = "(" + "|".join(escaped) + ")"
    parts = re.split(pattern, text)
    
    result = []
    special_set = set(special_tokens)
    
    for part in parts:
        if not part:
            continue
        if part in special_set:
            if keep_special_token:
                result.append(part)
                continue
            
            else:
                continue
        
        result.append(part)
    
    return result

def count_freq(pre_words):
    pre_count = Counter()
    pair_to_word = defaultdict(set)
    
    for word, frq in pre_words.items():
        for i in range(len(word) - 1):
            pre_count[(word[i], word[i+1])] += frq
            pair_to_word[(word[i], word[i+1])].add(word)
        
    
    return  pre_count, pair_to_word   
        
    
    
    
def merge_pair(best_pair, word):
    new_pair = best_pair[0] + best_pair[1]
    result = []
    i = 0
    while i < len(word):
        
        if i < len(word) -1:
            if word[i] == best_pair[0] and word[i+1] == best_pair[1] and i < len(word) - 1:
                result.append(new_pair)
                i = i + 2
            else:
                result.append(word[i])
                i = i + 1
        
        else:
            result.append(word[i])
            i = i+1
        
    return tuple(result)    
    
    

def find_best_pair(pre_count):
    if not pre_count:
        raise ValueError("pre_count must be non-empty")
    return max(pre_count.items(), key=lambda item: (item[1], item[0]))[0]

def add_word_contribution(pre_count, word, freq, pair_to_word):
    for i in range(len(word) - 1):
        p = (word[i], word[i+1])
        pre_count[p] += freq
        pair_to_word[p].add(word)
        

def remoev_word_contribution(pre_count, word, freq, pair_to_word):
    for i in range(len(word) - 1):
        p = (word[i], word[i+1])
        pre_count[p] -= freq
        if pre_count[p] == 0:
            del pre_count[p]

        pair_to_word[p].discard(word)
        if not pair_to_word[p]:
            del pair_to_word[p]

PRETOKEN_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):

        self.vocab = vocab
        self.inverse_vocab = {token: token_id for token_id, token in self.vocab.items()}
        self.merge_ranks = {pair: rank for rank, pair in enumerate(merges)}
        self.special_tokens = special_tokens or []
        self.cache = {}
        

    def _encode_pretoken(self, pretoken: str) -> list[int]:
        
        pre_token_bytes = pretoken.encode("utf8")
        if pre_token_bytes in self.cache:
            return self.cache[pre_token_bytes]
        tokens = tuple(bytes([b]) for b in pre_token_bytes)

        
        while True:
            best_pair = None
            best_rank = float("inf")
            if len(tokens) < 2: 
                break
            
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i+1])
                rank = self.merge_ranks.get(pair, float("inf"))
                if rank < best_rank:
                    best_rank = rank
                    best_pair = pair
            
            if best_pair is None or best_rank == float("inf"):
                break
                
            tokens = merge_pair(best_pair = best_pair, word = tokens)
        ids =[self.inverse_vocab[token] for token in tokens]
        self.cache[pre_token_bytes] = ids

        return ids
                




    def encode(self, text: str) -> list[int]:
        token_ids = []
        parts = seperate_sentence(self.special_tokens, text, keep_special_token=True)
        special_token_set = set(self.special_tokens)
        for part in parts:
            if part in special_token_set:
                token_ids.append(self.inverse_vocab[part.encode("utf-8")])
            else:
                for match in re.finditer(PRETOKEN_PATTERN, part):
                    pretoken = match.group(0)
                    token_ids.extend(self._encode_pretoken(pretoken))

        return token_ids

        # for match in re.finditer(PRETOKEN_PATTERN, text):
        #     pretoken = match.group(0)
        #     token_ids.extend(self._encode_pretoken(pretoken))

        # return token_ids

    def decode(self, ids: list[int]) -> str:
        token_bytes = b"".join(self.vocab[token_id] for token_id in ids)
        return token_bytes.decode("utf-8", errors="replace")
    
        # token_bytes = b"".join(self.vocab[token_id] for token_id in ids)
        # return token_bytes.decode("utf-8", errors="replace")

    def encode_iterable(self, iterable):
        for text in iterable:
            yield from self.encode(text)

    @classmethod
    def from_files(cls, vocab_path: str, merges_path: str, config_path: str):
        with open(vocab_path, "r", encoding="utf-8") as f:
            special_vocab = json.load(f)
        vocab = {int(token_id): bytes.fromhex(token_hex) for token_id, token_hex in special_vocab.items()}
        
        with open(merges_path, "r", encoding="utf-8") as f:
            special_merges = json.load(f)
        merges = [(bytes.fromhex(left_hex), bytes.fromhex(right_hex)) for left_hex, right_hex in special_merges]

        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        special_tokens = config.get("special_tokens", [])

        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)
