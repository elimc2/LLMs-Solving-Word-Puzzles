from datasets import load_from_disk

ds = load_from_disk("word_puzzle_cot")    # → a DatasetDict(train/validation)

ds.push_to_hub(
  "eli-equals-mc-2/WordPuzzleDataset",
  private=True
)
