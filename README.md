# Byte Pair Encoding (BPE) Tokenizer

This project implements a Byte Pair Encoding (BPE) tokenizer from scratch in Python. The BPE tokenizer is trained on a corpus to create subword units and can be used to tokenize new text using the learned subword vocabulary.

## Key Implementation Points

1. **Pre-tokenization**:
   - Text is normalized by converting to lowercase and removing non-alphabetical characters.
   - Words are split into characters initially for BPE training.

2. **Training**:
   - The `BPETokenizerTrainer` class handles training the tokenizer by:
     - Building word frequencies from the corpus.
     - Iteratively merging the most frequent token pairs to create subword units.
   - The number of subword units (vocabulary size) is configurable via `vocab_size`.

3. **Tokenization**:
   - Uses learned merge rules to tokenize new input text into subword units.
   - Merges character pairs iteratively based on the learned rules until no further merges are possible.

4. **Merge Rules**:
   - Learned during training and stored as a dictionary.
   - Can be serialized and reused for tokenizing new data.

5. **Dataset**:
   - Trains on the **English Wikipedia** dataset provided by the `lucadiliello/english_wikipedia` dataset on Hugging Face.

## How It Works

1. **Training the Tokenizer**:
   - The `BPETokenizerTrainer` processes a corpus to learn merge rules for subword tokenization.
   - Training uses the BPE algorithm to iteratively merge the most frequent token pairs.
   - A progress bar (`tqdm`) tracks vocabulary growth during training.

2. **Tokenizing New Text**:
   - Pre-tokenized words are split into characters.
   - Learned merge rules are applied to convert characters into subword units.

## Dependencies

The following dependencies are required:

- `datasets` (for loading the English Wikipedia dataset)
- `tqdm` (for progress tracking)

Install dependencies with:
```bash
pip install datasets tqdm
```
