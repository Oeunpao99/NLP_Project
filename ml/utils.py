"""
Utility functions for NER processing
"""

import re
from typing import List, Dict
import unicodedata

def normalize_khmer_text(text: str) -> str:
    """Normalize Khmer text"""
    # Normalize Unicode
    text = unicodedata.normalize('NFC', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def extract_entities(ner_results: List[Dict[str, str]]) -> Dict[str, List[str]]:
    """Extract entities from NER results.

    Supports both BIO labels (B-/I-) and flat labels (e.g., 'PER', 'ORG').
    If labels are BIO-style the original logic is used. For flat labels, consecutive
    non-'O' labels with the same type are merged into a single entity.
    """
    entities = {}
    current_entity = None
    current_tokens = []

    # Quick probe: are labels BIO-style?
    is_bio = any(result.get("label", "").startswith(("B-", "I-")) for result in ner_results)

    if is_bio:
        # Original BIO handling
        for result in ner_results:
            token = result["token"]
            label = result["label"]

            if label.startswith("B-"):
                # Save previous entity
                if current_entity:
                    entities.setdefault(current_entity, []).append(" ".join(current_tokens))

                # Start new entity
                entity_type = label.split("-", 1)[1]
                current_entity = entity_type
                current_tokens = [token]

            elif label.startswith("I-"):
                # Continue current entity
                if current_entity:
                    current_tokens.append(token)

            else:
                # Save previous entity
                if current_entity:
                    entities.setdefault(current_entity, []).append(" ".join(current_tokens))
                    current_entity = None
                    current_tokens = []

        # Save last entity if exists
        if current_entity:
            entities.setdefault(current_entity, []).append(" ".join(current_tokens))

    else:
        # Flat-label handling (e.g., 'PER', 'ORG', 'O')
        for result in ner_results:
            token = result.get("token")
            label = result.get("label")

            if label is None or label == "O":
                if current_entity:
                    entities.setdefault(current_entity, []).append(" ".join(current_tokens))
                    current_entity = None
                    current_tokens = []
                continue

            # Treat any non-'O' label as entity type (merge consecutive same-type tokens)
            entity_type = label.split("-")[-1] if "-" in label else label

            if current_entity == entity_type:
                current_tokens.append(token)
            else:
                if current_entity:
                    entities.setdefault(current_entity, []).append(" ".join(current_tokens))
                current_entity = entity_type
                current_tokens = [token]

        if current_entity:
            entities.setdefault(current_entity, []).append(" ".join(current_tokens))

    return entities

def format_ner_output(ner_results: List[Dict[str, str]], format: str = "html") -> str:
    """Format NER results in different formats"""
    if format == "html":
        return _format_html(ner_results)
    elif format == "json":
        return _format_json(ner_results)
    elif format == "text":
        return _format_text(ner_results)
    else:
        return _format_json(ner_results)

def _format_html(ner_results: List[Dict[str, str]]) -> str:
    """Format NER results as HTML with colored spans"""
    html_parts = []
    colors = {
        "PER": "#FF9999",      # Person - Red
        "ORG": "#99CCFF",      # Organization - Blue
        "LOC": "#99FF99",      # Location - Green
        "MISC": "#FFCC99",     # Miscellaneous - Orange
    }
    
    for result in ner_results:
        token = result["token"]
        entity_type = result["entity_type"]
        
        if entity_type != "O":
            color = colors.get(entity_type, "#CCCCCC")
            html_parts.append(
                f'<span class="entity" style="background-color: {color}; padding: 2px 4px; margin: 0 1px; border-radius: 3px;" '
                f'data-entity="{entity_type}">{token}</span>'
            )
        else:
            html_parts.append(token)
    
    return " ".join(html_parts)

def _format_json(ner_results: List[Dict[str, str]]) -> Dict:
    """Format NER results as JSON"""
    return {
        "tokens": [r["token"] for r in ner_results],
        "labels": [r["label"] for r in ner_results],
        "entities": extract_entities(ner_results)
    }

def _format_text(ner_results: List[Dict[str, str]]) -> str:
    """Format NER results as plain text with labels"""
    lines = []
    for result in ner_results:
        lines.append(f"{result['token']}\t{result['label']}")
    return "\n".join(lines)