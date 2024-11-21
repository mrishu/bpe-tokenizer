from collections import defaultdict
from itertools import chain
import re
from datasets import load_dataset
from tqdm import tqdm
import pickle


# This produces a stream of normalized words (tokens) from the corpus
# These are just word-level tokens, not the BPE tokens
def pre_tokenize(text: str):
    text = re.sub(
        "[^A-Za-z]+", " ", text
    ).lower()  # Replace non-alphabets with space and lowercase
    tokens = text.split()  # Split by whitespace into tokens
    return tokens


class BPETokenizerTrainer:
    def __init__(self, vocab_size=500, vocab=set()) -> None:
        # Save the target vocabulary size for future use
        self.vocab_size = vocab_size
        self.vocab = vocab  # stores vocabulary of tokens
        self.word_freqs = defaultdict(int)  # dictionary to store word frequencies
        self.splits = defaultdict(list)  # dictionary to store word splits
        self.merge_rules = defaultdict(str)  # Stores the merge rules

    def preprocess(self, corpus):
        # Initialise word frequencies, splits and vocab using documents from corpus
        for doc in corpus:
            words = pre_tokenize(doc)
            for word in words:
                self.word_freqs[word] += 1
                if word not in self.splits:
                    split = list(word)
                    self.splits[word] = split
                    self.vocab.update(split)

    # Given a pair of tokens (a, b), it performs merge of consecutive a and b in each split of self.splits
    def _merge_pair(self, a, b):
        for word in self.word_freqs.keys():
            split = self.splits[word]
            if len(split) == 1:
                continue
            i = 0
            while i < len(split) - 1:
                if split[i] == a and split[i + 1] == b:
                    split = split[:i] + [a + b] + split[i + 2 :]
                else:
                    i += 1
            self.splits[word] = split

    # This returns a dictionary of all current pairs (across all splits) and their frequencies
    def _compute_pair_freqs(self):
        pair_freqs = defaultdict(int)
        for word, freq in self.word_freqs.items():
            split = self.splits[word]
            if len(split) == 1:
                continue
            for i in range(len(split) - 1):
                pair = (split[i], split[i + 1])
                pair_freqs[pair] += freq
        return pair_freqs

    # This trains the BPE tokenizer
    def train(self):
        # Update splits, merge_rules and vocab using BPE tokenization algorithm
        progress_bar = tqdm(
            total=self.vocab_size, desc="Vocabulary Size"
        )  # Initialize the progress bar
        progress_bar.update(len(self.vocab))
        while len(self.vocab) < self.vocab_size:
            pair_freqs = self._compute_pair_freqs()
            if not pair_freqs:
                break  # stop if no pairs found
            best_pair = ""
            max_freq = None
            for pair, freq in pair_freqs.items():
                if max_freq is None or max_freq < freq:
                    best_pair = pair
                    max_freq = freq
            self._merge_pair(best_pair[0], best_pair[1])
            self.merge_rules[best_pair] = best_pair[0] + best_pair[1]
            self.vocab.add(best_pair[0] + best_pair[1])
            progress_bar.update(1)
        progress_bar.close()
        return self.merge_rules


def bpetokenizer(text, merge_rules):
    # Pre-tokenize the input text into words
    words = pre_tokenize(text)
    # Split each word into a list of characters
    splits = [list(word) for word in words]
    # Continue merging until no more changes occur
    merging = True
    while merging:
        merging = False
        for pair, merge in merge_rules.items():
            for idx, split in enumerate(splits):
                i = 0
                while i < len(split) - 1:
                    if split[i] == pair[0] and split[i + 1] == pair[1]:
                        # Replace the pair with the merged token
                        split = split[:i] + [merge] + split[i + 2 :]
                        merging = True  # A merge occurred, so continue the loop
                    else:
                        i += 1
                splits[idx] = split
    # Flatten the list of token lists into a single list of tokens
    return list(chain.from_iterable(splits))


if __name__ == "__main__":
    MAX_DOCS = 10000
    ds = load_dataset("lucadiliello/english_wikipedia", split="train")

    def documents_from_dataset(ds):
        docno = 0
        for doc in tqdm(ds):
            yield doc["maintext"]
            docno += 1
            if docno >= MAX_DOCS:
                break

    corpus = documents_from_dataset(ds)

    bpe_trainer = BPETokenizerTrainer()

    print(f"\nBuilding word frequencies... (Processing {MAX_DOCS} documents)")
    bpe_trainer.preprocess(corpus)

    print("\nTraining...")
    merge_rules = bpe_trainer.train()
    pickle.dump(merge_rules, open("bpe_trained_merge_rules.pkl", "wb"))

    # Comment the above if merge_rules is already saved
    merge_rules = pickle.load(open("bpe_trained_merge_rules.pkl", "rb"))

    inp = input("Enter a string to tokenize: ")
    print(bpetokenizer(inp, merge_rules))
