from datasets import load_dataset

WIKITEXT_DATASET = "wikitext"
WIKITEXT_CONFIG = "wikitext-2-raw-v1"
MIN_WIKI_WORDS = 32


def normalize_wiki_text(text):
    """Collapse whitespace and trim empty WikiText rows."""
    return " ".join(text.split()).strip()


def split_wiki_text(text, min_prompt_words=MIN_WIKI_WORDS):
    """Split a WikiText row into prompt/completion halves for calibration."""
    normalized = normalize_wiki_text(text)
    words = normalized.split()
    if len(words) < 2:
        raise ValueError("WikiText sample must contain at least two words.")

    split_idx = max(min_prompt_words, len(words) // 2)
    split_idx = min(split_idx, len(words) - 1)
    prompt = " ".join(words[:split_idx])
    completion = " ".join(words[split_idx:])
    return prompt, completion


def wiki_transform(batch):
    """Transform WikiText rows into text-only calibration examples."""
    processed_texts = []
    processed_visuals = []
    processed_answers = []
    processed_full_answers = []
    processed_org_texts = []

    for raw_text in batch["text"]:
        prompt, completion = split_wiki_text(raw_text)
        instruction = "Continue the following Wikipedia passage:\n\n"
        processed_visuals.append(None)
        processed_texts.append(instruction + prompt)
        processed_org_texts.append(prompt)
        processed_answers.append(completion)
        processed_full_answers.append(completion)

    return {
        "model_input_text": processed_texts,
        "model_input_visual": processed_visuals,
        "model_input_answer": processed_answers,
        "model_input_full_answer": processed_full_answers,
        "model_input_org_text": processed_org_texts,
    }


def load_wiki_dataset(split="train"):
    """Load non-empty WikiText rows suitable for text-only calibration."""
    data = load_dataset(WIKITEXT_DATASET, WIKITEXT_CONFIG, split=split)
    data = data.filter(
        lambda row: len(normalize_wiki_text(row["text"]).split()) > MIN_WIKI_WORDS
    )
    data.set_transform(wiki_transform)
    return data
