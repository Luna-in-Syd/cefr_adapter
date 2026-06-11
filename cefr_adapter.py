# COMP6713 Assignment 1 - CEFR Vocabulary Adapter
# rename this file to your zID before submission

import re
import math
import collections
from pathlib import Path

import pandas as pd
import spacy
import gensim.downloader as gensim_api
import pyinflect  # noqa: this patches spaCy tokens with a .inflect() method


CEFR_LEVELS = ["A1", "A2", "B1", "B2", "C1", "C2"]
LEVEL_INDEX  = {lvl: i for i, lvl in enumerate(CEFR_LEVELS)}  # A1=0, C2=5

# hardcoded stopwords so don't need to pull in NLTK just for this
STOPWORDS = set(
    "i me my myself we our ours ourselves you your yours yourself yourselves "
    "he him his himself she her hers herself it its itself they them their "
    "theirs themselves what which who whom this that these those am is are "
    "was were be been being have has had having do does did doing a an the "
    "and but if or because as until while of at by for with about against "
    "between into through during before after above below to from up down "
    "in out on off over under again further then once here there when where "
    "why how all both each few more most other some such no nor not only own "
    "same so than too very s t can will just said also etc".split()
)


# Data loading

def load_training_data(path="data.csv"):
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError("data.csv not found.")
    df = pd.read_csv(file_path)
    if not {"text", "cefr_level"}.issubset(df.columns):
        raise ValueError("data.csv must contain columns: text, cefr_level")
    return df


# Step 1: Build a word -> CEFR difficulty level map

def _simple_tokenize(text):
    # just grab alphabetic words, lowercase, nothing fancy
    return re.findall(r"[a-zA-Z]+", text.lower())


def build_vocab_difficulty(df):
    """
    Goes through the corpus and assigns each word a CEFR level based on
    where it shows up most often (relative to total word count per level).
    So "purchase" appearing a lot in C1 texts gets labelled C1, while
    "buy" appearing mostly in A2 texts gets labelled A2.
    Words seen fewer than twice are dropped since they are too noisy to trust.
    """
    level_counts = {lvl: collections.Counter() for lvl in CEFR_LEVELS}
    level_totals = {lvl: 0 for lvl in CEFR_LEVELS}

    for _, row in df.iterrows():
        lvl = row["cefr_level"]
        if lvl not in CEFR_LEVELS:
            continue
        # content words only: drop stopwords and anything 2 chars or shorter
        tokens = [
            t for t in _simple_tokenize(str(row["text"]))
            if t not in STOPWORDS and len(t) > 2
        ]
        level_counts[lvl].update(tokens)
        level_totals[lvl] += len(tokens)

    # pool all words that appear at least twice across any level
    all_words = set()
    for lvl in CEFR_LEVELS:
        all_words.update(w for w, c in level_counts[lvl].items() if c >= 2)

    vocab_dict = {}
    for word in all_words:
        # relative freq = how common is this word as a fraction of all words at that level
        rel_freq = {
            lvl: level_counts[lvl][word] / (level_totals[lvl] + 1)
            for lvl in CEFR_LEVELS
        }
        best_level = max(CEFR_LEVELS, key=lambda l: rel_freq[l])
        vocab_dict[word] = {
            "level":     best_level,
            "level_idx": LEVEL_INDEX[best_level],
            "rel_freq":  rel_freq,
        }
    return vocab_dict

def load_external_cefr(path="cefr_wordlist.csv"):
    df = pd.read_csv(path)
    external = {}
    for _, row in df.iterrows():
        word  = str(row["headword"]).lower().strip()
        level = str(row["CEFR"]).upper().strip()
        if level in CEFR_LEVELS:
            external[word] = {
                "level":     level,
                "level_idx": LEVEL_INDEX[level],
                "rel_freq":  {lvl: 0 for lvl in CEFR_LEVELS},
            }
    return external

def get_word_level(word, vocab_dict, default="B1"):
    # unknown words get B1 as a safe middle-ground default
    entry = vocab_dict.get(word.lower())
    return entry["level"] if entry else default


def needs_replacement(word, target_level, vocab_dict):
    """
    Decides if a word is "wrong" for the target level.
    When simplifying: swap out anything harder than the target.
    When complexifying: only bother if the word is 2+ levels too easy,
    no point swapping "walk" for "stroll" when going from A2 to B1.
    """
    word_idx   = LEVEL_INDEX[get_word_level(word, vocab_dict)]
    target_idx = LEVEL_INDEX[target_level]
    return word_idx > target_idx or word_idx < target_idx - 2


# Step 2: POS tagging

# only these four POS types are worth replacing
CONTENT_POS = {"NOUN", "VERB", "ADJ", "ADV"}


def tag_sentence(sentence, nlp):
    """
    Runs spaCy on the input and returns a list of token info dicts.
    We store whitespace separately so we can glue the sentence back together
    at the end without mangling punctuation or spacing.
    """
    doc = nlp(sentence)
    tokens = []
    for tok in doc:
        clean = re.sub(r"[^a-zA-Z]", "", tok.text).lower()
        is_content = (
            tok.pos_ in CONTENT_POS
            and clean not in STOPWORDS
            and len(clean) > 2
            and clean.isalpha()  # skip tokens with digits or symbols mixed in
        )
        tokens.append({
            "text":       tok.text,
            "lower":      clean,
            "lemma":      tok.lemma_.lower(),
            "pos":        tok.pos_,
            "tag":        tok.tag_,       # Penn Treebank tag e.g. "VBD", needed by pyinflect
            "is_content": is_content,
            "whitespace": tok.whitespace_,
        })
    return tokens


def find_words_to_replace(tokens, target_level, vocab_dict):
    """Returns indices of the tokens that need swapping out."""
    return [
        i for i, tok in enumerate(tokens)
        if tok["is_content"] and needs_replacement(tok["lemma"], target_level, vocab_dict)
    ]


# Step 3: Candidate generation via GloVe

# maps spaCy POS to Penn Treebank prefixes so we only swap nouns for nouns, etc.
POS_TAG_PREFIX = {
    "NOUN": ("NN",),
    "VERB": ("VB",),
    "ADJ":  ("JJ",),
    "ADV":  ("RB",),
}


def find_candidates(lemma, pos, target_level, vocab_dict, word_vectors, nlp,
                    topn=30, max_candidates=10):
    """
    Asks GloVe "what words are closest to this one?" then filters the results
    down to words that actually make sense as replacements:
      - must be in our CEFR vocab dict
      - CEFR level within +-1 of target (exact match ideal, neighbour ok)
      - same part of speech as the original word

    Tries cosine similarity >= 0.65 first. If that returns nothing,
    relaxes to >= 0.5. Better to have some candidate than none.
    """
    if lemma not in word_vectors:
        return []

    similar    = word_vectors.most_similar(lemma, topn=topn)
    target_idx = LEVEL_INDEX[target_level]
    allowed_prefixes = POS_TAG_PREFIX.get(pos, ())

    def _search(threshold):
        found = []
        for word, score in similar:
            if score < threshold:
                break  # results are sorted descending, so we can stop early
            if not word.isalpha() or word == lemma:
                continue
            if word not in vocab_dict:
                continue
            if abs(vocab_dict[word]["level_idx"] - target_idx) > 1:
                continue
            if allowed_prefixes:
                cand_tag = nlp(word)[0].tag_ if word in word_vectors else ""
                if not any(cand_tag.startswith(p) for p in allowed_prefixes):
                    continue
            found.append(word)
            if len(found) >= max_candidates:
                break
        return found

    candidates = _search(0.65)
    if not candidates:
        candidates = _search(0.5)
    return candidates


# Step 4: Bigram LM for fluency ranking

def build_bigram_lm(df):
    """Counts all bigrams and unigrams in the corpus. Pretty straightforward."""
    bigram_counts  = collections.Counter()
    unigram_counts = collections.Counter()
    for text in df["text"].astype(str):
        tokens = _simple_tokenize(text)
        unigram_counts.update(tokens)
        for w1, w2 in zip(tokens, tokens[1:]):
            bigram_counts[(w1, w2)] += 1
    return bigram_counts, unigram_counts


def bigram_score(candidate, idx, output_tokens, bigram_counts, unigram_counts):
    """
    Scores how well a candidate fits at position idx by looking at two bigrams:
      P(candidate | previous_word) and P(next_word | candidate)
    Uses Laplace (add-1) smoothing to handle unseen bigrams gracefully.
    Higher score = word fits more naturally in context.
    """
    V     = len(unigram_counts)  # vocab size for smoothing denominator
    score = 0.0
    cand  = candidate.lower()

    if idx > 0:
        prev = re.sub(r"[^a-zA-Z]", "", output_tokens[idx - 1]).lower()
        if prev:
            prob   = (bigram_counts.get((prev, cand), 0) + 1) / (unigram_counts.get(prev, 0) + V)
            score += math.log(prob)

    if idx < len(output_tokens) - 1:
        nxt  = re.sub(r"[^a-zA-Z]", "", output_tokens[idx + 1]).lower()
        if nxt:
            prob   = (bigram_counts.get((cand, nxt), 0) + 1) / (unigram_counts.get(cand, 0) + V)
            score += math.log(prob)

    return score


def rank_by_lm(candidates, idx, output_tokens, bigram_counts, unigram_counts):
    """Picks whichever candidate scores highest under the bigram model."""
    return max(
        candidates,
        key=lambda c: bigram_score(c, idx, output_tokens, bigram_counts, unigram_counts)
    )


# Step 5: Inflection matching

def inflect_word(base_word, original_tok, nlp):
    """
    Takes the base/lemma form of a replacement and inflects it to match
    the original token's grammatical form. For example:
      base="use", original tag="VBD" (past tense) -> returns "used"
    Falls back to the uninflected base if pyinflect can't handle the tag.
    """
    tag = original_tok["tag"]
    doc = nlp(base_word)
    if not doc:
        return base_word
    inflected = doc[0]._.inflect(tag)
    return inflected if inflected else base_word


# Load everything once at startup

print("Loading training data...")
_df   = load_training_data("data.csv")
VOCAB = build_vocab_difficulty(_df)
print(f"  vocab built: {len(VOCAB)} words")
if Path("cefr_wordlist.csv").exists():
    external = load_external_cefr("cefr_wordlist.csv")
    VOCAB.update(external)
    print(f"  external wordlist merged: {len(external)} entries")

print("Loading spaCy model...")
NLP = spacy.load("en_core_web_sm")

# glove-wiki-gigaword-100: 100-dimensional, ~65 MB, cached after first download
print("Loading GloVe vectors (first run will download ~65 MB)...")
WORD_VECTORS = gensim_api.load("glove-wiki-gigaword-100")

print("Building bigram LM...")
BIGRAM_COUNTS, UNIGRAM_COUNTS = build_bigram_lm(_df)
print("Ready.\n")


# Main entry point

def transform_sentence(sentence, source_level, target_level):
    """
    Rewrites a sentence so its vocabulary fits the target CEFR level.
    Tries to change as few words as possible to preserve the original meaning.

    Pipeline overview:
      1. Tag the sentence with spaCy (POS + lemma)
      2. Find content words that are too hard or too easy for the target level
      3. For each flagged word, retrieve GloVe neighbours at the right CEFR level
      4. Rank candidates by bigram LM score (contextual fluency)
      5. Inflect the winner to match the original word's grammatical form
      6. Patch up a/an agreement if the replacement starts with a vowel/consonant
    """
    if source_level not in CEFR_LEVELS:
        raise ValueError(f"Unknown source level: {source_level}")
    if target_level not in CEFR_LEVELS:
        raise ValueError(f"Unknown target level: {target_level}")

    if source_level == target_level:
        return sentence

    tokens        = tag_sentence(sentence, NLP)
    to_replace    = find_words_to_replace(tokens, target_level, VOCAB)
    output_tokens = [tok["text"] for tok in tokens]

    for idx in to_replace:
        tok   = tokens[idx]
        lemma = tok["lemma"]
        pos   = tok["pos"]

        candidates = find_candidates(lemma, pos, target_level, VOCAB, WORD_VECTORS, NLP)
        if not candidates:
            continue  # GloVe drew a blank, leave the word alone

        best        = rank_by_lm(candidates, idx, output_tokens, BIGRAM_COUNTS, UNIGRAM_COUNTS)
        replacement = inflect_word(best, tok, NLP)

        # keep the same capitalisation as the original (e.g. sentence-initial caps)
        if tok["text"][0].isupper():
            replacement = replacement.capitalize()

        output_tokens[idx] = replacement

        # fix "a"/"an" agreement for the article before the replacement
        if idx > 0:
            prev = output_tokens[idx - 1].lower().rstrip(".,;:")
            if prev in ("a", "an"):
                first       = replacement.lstrip().lower()[0] if replacement.strip() else ""
                new_article = "an" if first in "aeiou" else "a"
                if tokens[idx - 1]["text"][0].isupper():
                    new_article = new_article.capitalize()
                output_tokens[idx - 1] = new_article

    # stitch the sentence back together using spaCy's original whitespace info
    result = "".join(out + tok["whitespace"] for tok, out in zip(tokens, output_tokens))
    return result.strip()


# sanity check, run a few examples when executing this file directly
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CEFR Vocabulary Adapter")
    parser.add_argument("--text",   required=True,  help="Input sentence")
    parser.add_argument("--from",   dest="src",     required=True, choices=CEFR_LEVELS)
    parser.add_argument("--to",     dest="tgt",     required=True, choices=CEFR_LEVELS)
    args = parser.parse_args()

    result = transform_sentence(args.text, args.src, args.tgt)
    print(f"Original : {args.text}")
    print(f"Adapted  : {result}")