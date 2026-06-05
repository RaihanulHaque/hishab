import json
import re
from collections import Counter
from pathlib import Path

dataset_file = str(Path(__file__).parent.parent / "datasets" / "synthetic_dataset.jsonl")

# Extract all words from Banglish user messages only
all_words = []
banglish_count = 0

with open(dataset_file, "r") as f:
    for line in f:
        if line.strip():
            data = json.loads(line)
            # Filter by metadata.lang == "banglish"
            if data.get("metadata", {}).get("lang") == "banglish":
                banglish_count += 1
                for msg in data["messages"]:
                    if msg["role"] == "user":
                        # Extract all romanized words
                        words = re.findall(r'\b[a-zA-Z]+\b', msg["content"].lower())
                        all_words.extend(words)

print(f"Found {banglish_count} Banglish entries in dataset")
print(f"Total words extracted: {len(all_words)}")

# Common English words to exclude (these appear in Banglish but are English)
common_english = {
    # Articles, prepositions, conjunctions
    "the", "a", "an", "is", "are", "for", "and", "or", "from", "to", "of", 
    "in", "on", "at", "by", "with", "as", "it", "this", "that", "was", "were",
    "be", "been", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "must", "can", "i", "my", "me", "we",
    "our", "you", "your", "he", "she", "his", "her", "they", "their", "them",
    "what", "which", "who", "whom", "when", "where", "why", "how", "all",
    "each", "every", "both", "few", "more", "most", "other", "some", "such",
    "no", "not", "only", "same", "so", "than", "too", "very", "just", "but",
    "if", "because", "until", "while", "about", "after", "before", "above",
    "below", "between", "into", "through", "during", "out", "off", "over",
    "under", "again", "further", "then", "once", "here", "there", "any",
    # Financial English terms commonly used in Banglish
    "bill", "payment", "paid", "received", "total", "amount", "money",
    "charge", "fee", "cost", "price", "bank", "account", "transfer",
    "cash", "credit", "loan", "income", "expense", "salary", "rent",
    "internet", "mobile", "phone", "recharge", "gas", "electricity",
    "water", "transport", "bus", "taxi", "train", "food", "lunch",
    "dinner", "breakfast", "restaurant", "hotel", "hospital", "doctor",
    "medicine", "medical", "gift", "shopping", "market", "grocery",
    "tuition", "school", "college", "university", "exam", "ticket",
    "cinema", "movie", "entertainment", "project", "service", "services",
    "client", "office", "work", "job", "monthly", "daily", "return",
    "refund", "lost", "send", "sent", "via", "online", "pack", "package",
    "new", "old", "first", "last", "next", "previous", "also", "plus",
    "included", "including", "around", "approximately", "need", "emi",
    "ice", "cream", "shirt", "pant", "uniform", "stationery", "items",
}

# Count word frequencies
word_counts = Counter(all_words)

# Filter for Banglish words (frequent, not common English, min length 2)
banglish_candidates = [
    (word, count) for word, count in word_counts.most_common(500)
    if word not in common_english and count >= 2 and len(word) > 1
]

print("\n" + "=" * 60)
print("TOP BANGLISH TOKENS (sorted by frequency)")
print("=" * 60)
for word, count in banglish_candidates[:100]:
    print(f"{word:20} : {count}")

print("\n" + "=" * 60)
print("CATEGORIZED BANGLISH TOKENS")
print("=" * 60)

# Group by category based on common Banglish patterns
verb_tokens = []
noun_tokens = []
connector_tokens = []
other_tokens = []

for word, count in banglish_candidates:
    # Common Banglish verb endings
    if word.endswith(('lam', 'lum', 'chi', 'si', 'che', 'lo', 'lum', 'echi', 'achi', 'ilo', 'ese', 'alam')):
        verb_tokens.append((word, count))
    # Common Banglish connectors/postpositions
    elif word in ['ar', 'r', 'theke', 'jonno', 'diye', 'kore', 'ta', 'te', 'er', 'ke', 'e', 'o', 'ba']:
        connector_tokens.append((word, count))
    # Time/quantity words
    elif word in ['taka', 'tk', 'hajar', 'lakh', 'sho', 'ajke', 'gotokal', 'kal', 'mash', 'din', 'rate', 'shob', 'miliye', 'modhye', 'kore']:
        noun_tokens.append((word, count))
    else:
        other_tokens.append((word, count))

print("\n--- Verb-like tokens (action words) ---")
for word, count in verb_tokens[:25]:
    print(f"  {word:20} : {count}")

print("\n--- Connector/postposition tokens ---")
for word, count in connector_tokens[:15]:
    print(f"  {word:20} : {count}")

print("\n--- Noun/time/quantity tokens ---")
for word, count in noun_tokens[:20]:
    print(f"  {word:20} : {count}")

print("\n--- Other common tokens ---")
for word, count in other_tokens[:40]:
    print(f"  {word:20} : {count}")

# Final recommended list - prioritize by frequency and importance
all_categorized = verb_tokens + connector_tokens + noun_tokens + other_tokens
recommended_tokens = [word for word, _ in banglish_candidates]

print("\n" + "=" * 60)
print("COPY THIS LIST FOR TOKENIZER AUGMENTATION:")
print("=" * 60)
print(f"\nbanglish_tokens = {recommended_tokens}")

print("\n" + "=" * 60)
print("STATS")
print("=" * 60)
print(f"Total unique Banglish words found: {len(banglish_candidates)}")
print(f"Verb-like tokens: {len(verb_tokens)}")
print(f"Connector tokens: {len(connector_tokens)}")
print(f"Noun/quantity tokens: {len(noun_tokens)}")
print(f"Other tokens: {len(other_tokens)}")