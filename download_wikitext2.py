from datasets import load_dataset

# Download WikiText-2
dataset = load_dataset("wikitext", "wikitext-2-v1")

# Access the splits
train = dataset['train']
validation = dataset['validation']
test = dataset['test']

# Save locally if you want
train.save_to_disk("./wikitext2_train")
validation.save_to_disk("./wikitext2_val")
test.save_to_disk("./wikitext2_test")

print(f"Train samples: {len(train)}")
print(f"Sample text: {train[0]['text'][:200]}")