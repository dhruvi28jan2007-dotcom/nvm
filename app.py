
import re
from collections import Counter

import numpy as np
import streamlit as st
import plotly.graph_objects as go

import gensim.downloader as api
import torch

from transformers import AutoTokenizer, AutoModel


# -----------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------

st.set_page_config(
    page_title="NLP Visualizer",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 NLP Visualizer")
st.write(
    "Explore tokenization, spelling similarity, "
    "static embeddings and contextual embeddings."
)


# -----------------------------------------
# DEFAULT TRAINING SENTENCES
# -----------------------------------------

DEFAULT_TRAINING = """The cat sat on the mat.
The dog sat on the rug.
The cat likes the dog.
The sun is hot today.
I like my cat and my dog.
The dog ran to the park."""


WORD_PATTERN = re.compile(r"[A-Za-z0-9']+")


# -----------------------------------------
# PART 1: CUSTOM TOKENIZER
# -----------------------------------------

def build_vocabulary(text):

    words = WORD_PATTERN.findall(text.lower())

    frequency = Counter(words)

    vocabulary = {"[UNK]": 0}

    for word, count in frequency.most_common():
        vocabulary[word] = len(vocabulary)

    # Character-level fallback tokens
    for char in "abcdefghijklmnopqrstuvwxyz":
        if char not in vocabulary:
            vocabulary[char] = len(vocabulary)

        piece = "##" + char

        if piece not in vocabulary:
            vocabulary[piece] = len(vocabulary)

    return vocabulary, frequency


def tokenize_sentence(sentence, vocabulary, preserved_oov_words=None):

    words = WORD_PATTERN.findall(sentence.lower())

    tokens = []
    token_ids = []

    for word in words:

        if word in vocabulary:

            tokens.append(word)
            token_ids.append(vocabulary[word])

        elif preserved_oov_words and word in preserved_oov_words:

            tokens.append(word)
            token_ids.append(vocabulary["[UNK]"])

        else:

            # Unknown word: split into characters
            for i, char in enumerate(word):

                piece = char if i == 0 else "##" + char

                if piece in vocabulary:
                    tokens.append(piece)
                    token_ids.append(vocabulary[piece])

                else:
                    tokens.append("[UNK]")
                    token_ids.append(0)

    return tokens, token_ids


# -----------------------------------------
# PART 2: SPELLING SIMILARITY
# -----------------------------------------

def levenshtein(a, b):

    previous = list(range(len(b) + 1))

    for i, char_a in enumerate(a):

        current = [i + 1]

        for j, char_b in enumerate(b):

            insert = current[j] + 1
            delete = previous[j + 1] + 1

            replace = previous[j] + (
                char_a != char_b
            )

            current.append(
                min(insert, delete, replace)
            )

        previous = current

    return previous[-1]


def spelling_similarity(a, b):

    distance = levenshtein(a.lower(), b.lower())

    longest = max(len(a), len(b))

    if longest == 0:
        return 1.0

    return 1 - distance / longest


# -----------------------------------------
# PART 3: LOAD EMBEDDING MODELS
# -----------------------------------------

@st.cache_resource
def load_models():

    # Static embedding model
    glove = api.load("glove-wiki-gigaword-50")

    # Contextual embedding model
    model_name = "distilbert-base-uncased"

    tokenizer = AutoTokenizer.from_pretrained(
        model_name
    )

    model = AutoModel.from_pretrained(
        model_name
    )

    model.eval()

    return glove, tokenizer, model


# -----------------------------------------
# COSINE SIMILARITY
# -----------------------------------------

def cosine_similarity(a, b):

    denominator = (
        np.linalg.norm(a) * np.linalg.norm(b)
    )

    if denominator == 0:
        return 0.0

    return float(np.dot(a, b) / denominator)


# -----------------------------------------
# STATIC EMBEDDINGS
# -----------------------------------------

def get_static_embedding(word, glove):

    word = word.lower()

    if word in glove:
        return glove[word]

    return None


# -----------------------------------------
# CONTEXTUAL EMBEDDINGS
# -----------------------------------------

def get_contextual_embeddings(
    sentence,
    tokenizer,
    model
):

    encoded = tokenizer(
        sentence,
        return_tensors="pt",
        return_offsets_mapping=True,
        truncation=True
    )

    offsets = encoded.pop(
        "offset_mapping"
    )[0].tolist()

    with torch.no_grad():

        output = model(**encoded)

    hidden_states = (
        output.last_hidden_state[0].numpy()
    )

    words = list(
        WORD_PATTERN.finditer(sentence)
    )

    result = []

    for word_match in words:

        start, end = word_match.span()

        vectors = []

        for index, (token_start, token_end) in enumerate(
            offsets
        ):

            if (
                token_start < end
                and token_end > start
                and token_end > token_start
            ):

                vectors.append(
                    hidden_states[index]
                )

        if vectors:

            word_vector = np.mean(
                vectors,
                axis=0
            )

            result.append(
                (word_match.group().lower(), word_vector)
            )

    return result


# -----------------------------------------
# SIDEBAR: TRAINING VOCABULARY
# -----------------------------------------

st.sidebar.header("Training vocabulary")

training_text = st.sidebar.text_area(
    "Enter training sentences",
    value=DEFAULT_TRAINING,
    height=220
)

vocabulary, frequency = build_vocabulary(
    training_text
)


# -----------------------------------------
# MAIN INPUT
# -----------------------------------------

st.header("Enter sentences to compare")

col1, col2 = st.columns(2)

with col1:

    sentence1 = st.text_input(
        "Sentence 1",
        "I have a bat"
    )

with col2:

    sentence2 = st.text_input(
        "Sentence 2",
        "I have a cat"
    )


analyze = st.button(
    "Analyze sentences",
    type="primary"
)


# -----------------------------------------
# RUN ANALYSIS
# -----------------------------------------

if analyze:

    if not sentence1.strip() or not sentence2.strip():

        st.warning("Please enter both sentences.")

    elif not training_text.strip():

        st.warning("Please enter training sentences.")

    else:

        # TOKENIZATION

        st.header("Part 1: Tokenizer")

        words1 = set(WORD_PATTERN.findall(sentence1.lower()))
        words2 = set(WORD_PATTERN.findall(sentence2.lower()))
        shared_oov_words = (words1 & words2) - vocabulary.keys()

        tokens1, ids1 = tokenize_sentence(
            sentence1,
            vocabulary,
            shared_oov_words
        )

        tokens2, ids2 = tokenize_sentence(
            sentence2,
            vocabulary,
            shared_oov_words
        )

        st.caption(
            "Words shared by both sentences stay whole even when they are "
            "missing from the training vocabulary; their ID is [UNK]. "
            "Other unknown words are split into character pieces."
        )

        st.subheader("Sentence 1 tokens")

        st.write(tokens1)
        st.write("Token IDs:", ids1)

        st.subheader("Sentence 2 tokens")

        st.write(tokens2)
        st.write("Token IDs:", ids2)

        st.write(
            "Vocabulary size:",
            len(vocabulary)
        )

        # VOCABULARY TABLE

        st.subheader("Vocabulary frequency")

        vocab_data = []

        for word, count in frequency.most_common():

            vocab_data.append({
                "Token": word,
                "ID": vocabulary[word],
                "Count": count
            })

        st.dataframe(
            vocab_data,
            use_container_width=True
        )

        # LOAD MODELS

        with st.spinner(
            "Loading embedding models. First download may take time..."
        ):

            glove, bert_tokenizer, bert_model = (
                load_models()
            )

        st.success("Embedding models loaded!")

        # GET CONTEXTUAL EMBEDDINGS

        contextual1 = get_contextual_embeddings(
            sentence1,
            bert_tokenizer,
            bert_model
        )

        contextual2 = get_contextual_embeddings(
            sentence2,
            bert_tokenizer,
            bert_model
        )

        # COMPARE WORDS BY POSITION

        st.header("Part 2: Word comparison")

        comparisons = []

        for i, (item1, item2) in enumerate(
            zip(contextual1, contextual2)
        ):

            word1, context_vector1 = item1
            word2, context_vector2 = item2

            static_vector1 = get_static_embedding(
                word1, glove
            )

            static_vector2 = get_static_embedding(
                word2, glove
            )

            spelling = spelling_similarity(
                word1, word2
            )

            if (
                static_vector1 is not None
                and static_vector2 is not None
            ):

                static_score = cosine_similarity(
                    static_vector1,
                    static_vector2
                )

                static_status = "Available"

            else:

                static_score = 0.0
                static_status = "OOV: vector unavailable"

            contextual_score = cosine_similarity(
                context_vector1,
                context_vector2
            )

            comparisons.append({

                "Word 1": word1,
                "Word 2": word2,
                "Exact Match": word1 == word2,

                "Spelling Similarity": round(
                    spelling, 3
                ),

                "Static Similarity": round(
                    static_score, 3
                ),

                "Contextual Similarity": round(
                    contextual_score, 3
                ),

                "Static Status": static_status

            })

        if not comparisons:

            st.warning("No comparable words found.")

        else:

            st.dataframe(
                comparisons,
                use_container_width=True
            )

            # ---------------------------------
            # PART 3: 3D VISUALIZATION
            # ---------------------------------

            st.header("Part 3: Embeddings in 3D")

            x = [
                item["Spelling Similarity"]
                for item in comparisons
            ]

            y = [
                item["Static Similarity"]
                for item in comparisons
            ]

            z = [
                item["Contextual Similarity"]
                for item in comparisons
            ]

            labels = [
                item["Word 1"] + " ↔ " + item["Word 2"]
                for item in comparisons
            ]

            fig = go.Figure()

            fig.add_trace(
                go.Scatter3d(

                    x=x,
                    y=y,
                    z=z,

                    mode="markers+text",

                    text=labels,

                    textposition="top center",

                    marker=dict(
                        size=7,
                        opacity=0.85
                    ),

                    hovertemplate=(
                        "Words: %{text}<br>"
                        "Spelling: %{x:.3f}<br>"
                        "Static: %{y:.3f}<br>"
                        "Contextual: %{z:.3f}"
                        "<extra></extra>"
                    )

                )
            )

            fig.update_layout(

                title="Spelling vs Static vs Contextual Similarity",

                scene=dict(

                    xaxis_title="Spelling Similarity",

                    yaxis_title="Static Embedding Similarity",

                    zaxis_title="Contextual Embedding Similarity"

                ),

                height=650

            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            st.info(
                "Each point represents a pair of words "
                "at the same position in the two sentences. "
                "Cosine similarity measures embedding similarity. "
                "The graph shows similarity, not the original "
                "high-dimensional embedding coordinates."
            )
