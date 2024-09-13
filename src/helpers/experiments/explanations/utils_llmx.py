"""This is the main logic for the LLM-x explanation."""

import json
import re
from typing import List, Optional
import random

import torch
import numpy as np
from nltk.stem import PorterStemmer


def prepare_prompt(
    inputs: List[str],
    target: int,
    softmax: float,
    softmax_perturb: float,
    subtask: str,
    class_labels: list,
    top_K: int = 3,
    magnitude_level: str = "",
) -> str:

    # Defined outside the function for global use and efficiency
    MODEL_SYNONYMS = ["n AI", "n artificial intelligence", " machine learning"]
    PERTURBATION_SYNONYMS = [
        f"'s weights and biases has been {magnitude_level}adversarially manipulated",
        f" has been {magnitude_level}poisoned with noise",
        f" has been {magnitude_level}reparameterized",
        f"'s weights have been {magnitude_level}manipulated",
        f"'s weights have been {magnitude_level}perturbed",
    ]
    CERTAINTY_SYNONYMS = ["certainty", "confidence"]
    CERTAINTY_VERB_SYNONYMS = ["certain", "confident"]

    # For display, join class labels and sample strings.
    label = class_labels[target]
    class_labels_str = ", ".join([f'"{label.upper()}"' for label in class_labels])
    certain_str = random.choice(CERTAINTY_SYNONYMS)
    certain_verb_str = random.choice(CERTAINTY_VERB_SYNONYMS)
    perturb_str = random.choice(PERTURBATION_SYNONYMS)

    # Generate prompt depending on version.
    subtask_singular = subtask
    if subtask == "sms":
        subtask += " messages"
        subtask_singular += " message"
    elif subtask == "sentence":
        subtask += "s' sentiment"

    # Get change in softmax.
    sub_softmax = softmax - softmax_perturb
    symbol = "less" if sub_softmax < 0.0 else "more"

    # Pre-prompt.
    prompt = f"Context: A{random.choice(MODEL_SYNONYMS)} model has been trained to classify {subtask} as {class_labels_str}. "
    prompt += f'With {softmax_perturb*100:.1f}% {certain_str}, the model classified the {subtask_singular}: "{inputs}" as "{label.upper()}". '
    if not magnitude_level == "None":
        prompt += f"Now, the model{perturb_str}."

    # Mid-prompt.
    if np.isclose(sub_softmax, 1.0, atol=0.01):
        prompt += f' The model is as {certain_verb_str} in its classication of "{class_labels[target].upper()}"'
    else:
        prompt += f' The model is {abs(sub_softmax)*100:.1f}% {symbol} {certain_verb_str} in its classication of "{class_labels[target].upper()}"'

    if not magnitude_level == "None":
        prompt += f", after the perturbation."

    # Post-prompt.
    prompt += f'\nQuestion: Please rank the {top_K} most important single words or symbols (tokens) that explains this new "{class_labels[target].upper()}" prediction. '
    prompt += f"Provide your ranking in JSON-parsable format "
    prompt += r"{1: 'most important token', 2: 'second most important token', 3: 'third most important token'}"
    prompt += f", without ANY additional markdown or text on the last line."

    return prompt


def normalise(text, stem: bool = False):
    stemmer = PorterStemmer()
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    if stem:
        text = " ".join([stemmer.stem(word) for word in text.split()])
    return text


def generate_explanation_from_tokens(
    parsed_outputs,
    inputs,
    llm_tokenizer,
    max_length,
    binarise: bool = False,
    stem: bool = False,
    verbose: bool = False,
):
    torch.cuda.empty_cache()
    # Normalize and tokenize input texts, then encode.
    normalised_input_texts = [normalise(text, stem) for text in inputs]
    input_ids = llm_tokenizer(
        normalised_input_texts,
        padding="max_length",
        truncation=False,
        return_tensors="pt",
        max_length=max_length,
    ).input_ids

    # Normalize and tokenize top-K tokens for each response, then encode.
    top_k_sets = [" ".join(list(response.values())) for response in parsed_outputs]
    normalised_top_k_texts = [normalise(top_k_set, stem) for top_k_set in top_k_sets]
    top_k_ids_list = llm_tokenizer(
        normalised_top_k_texts,
        padding=True,
        truncation=True,
        return_tensors="pt",  # CHECK THIS!
    ).input_ids

    # Get the PAD token ID from the tokenizer.
    pad_token_id = llm_tokenizer.pad_token_id

    # Prepare to mask input IDs: replace all tokens not in top-K with the PAD token ID.
    masked_input_ids = []
    nans = []
    for i, (input_id_tensor, response) in enumerate(zip(input_ids, parsed_outputs)):
        if response is {}:
            nans.append(i)
        top_k_ids = set(top_k_ids_list[i].tolist())
        if binarise:
            masked_row = [
                1 if input_id in top_k_ids and input_id != 0 else pad_token_id
                for input_id in input_id_tensor.tolist()
            ]
        else:
            masked_row = [
                input_id if input_id in top_k_ids and input_id != 0 else pad_token_id
                for input_id in input_id_tensor.tolist()
            ]
        masked_input_ids.append(masked_row)

    explanations = torch.tensor(masked_input_ids, dtype=torch.float)
    for i in nans:
        explanations[i] = float("nan")

    if verbose:
        print(parsed_outputs)
        for i, input_text in enumerate(inputs):
            print(f"\nInput: {input_text} -> {normalised_input_texts[i]}")
            print(f"Tokens: {top_k_sets[i]} -> {normalised_top_k_texts[i]}")
            print(f"Masked Input IDs: {masked_input_ids[i]}")
            print(f"Masked Token matches: {np.sum(masked_input_ids[i])}")

        for ix, x in enumerate(parsed_outputs):
            if x is {}:
                print(explanations[ix])

        print("explanations shape:", explanations.shape)

    torch.cuda.empty_cache()
    return explanations


def safe_llm_parse(llm_output, verbose: bool = False):
    # Improve regex to match JSON-like substrings more reliably
    # This regex assumes that JSON objects are bracketed by { and } and can span multiple lines.
    json_candidates = re.findall(r"\{.*?\}(?=\,|$)", llm_output, re.DOTALL)

    if not json_candidates:
        if verbose:
            print("No JSON-like string found. Returning an empty dict.")
        return {}

    # json_candidates = re.sub(r"\.\s*\}", "}", llm_output)
    for json_str in json_candidates:
        if "most important token" not in json_str:
            if verbose:
                print(json_str)
            try:
                return eval(str(json_str))
            except Exception as e:
                print(
                    f"Function 'safe_json_parse' failed with error: {e}. Returning an empty dict.\nJSON string: {str(json_str)}"
                )
                return {}
