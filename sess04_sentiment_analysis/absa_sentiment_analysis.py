"""
============================================================================================================
Python script to demonstrate ABSA (Abpect Based Sentiment Analysis)
============================================================================================================

<strong>WHAT IS ABSA?</strong>
<strong>-------------</strong>
<strong>Standard sentiment analysis assigns ONE sentiment to an entire sentence:</strong>
<strong>    "The battery is great but the screen is awful."  →  MIXED</strong>

<strong>ABSA assigns sentiment PER ASPECT (topic/feature):</strong>
<strong>    battery  →  POSITIVE</strong>
<strong>    screen   →  NEGATIVE</strong>

<strong>This gives far richer, more actionable insights — especially useful for</strong>
<strong>product reviews, customer feedback, and survey analysis.</strong>

<strong>HOW THIS DEMO WORKS</strong>
<strong>--------------------</strong>
<strong>  1. ASPECT EXTRACTION  — rule-based keyword matching across 9 product</strong>
<strong>                          categories (battery, screen, camera, etc.)</strong>
<strong>  2. CLAUSE SPLITTING   — the sentence is split at conjunctions (but, and,</strong>
<strong>                          however, …) so that each clause is scored in</strong>
<strong>                          isolation, preventing sentiments from bleeding</strong>
<strong>                          across aspects.</strong>
<strong>  3. SENTIMENT SCORING  — VADER (Valence Aware Dictionary and sEntiment</strong>
<strong>                          Reasoner) scores each clause.  VADER is purpose-</strong>
<strong>                          built for short, informal text and requires NO</strong>
<strong>                          model download or GPU.</strong>


Requirements:
    pip install vaderSentiment

Date:
    15th May 2026
"""
# ----------------------------------------------------------------------------------------------
# 0. Import the required modules
# ----------------------------------------------------------------------------------------------
from __future__ import annotations # Ensure this is the 1st import to avoid errors
import re, sys, textwrap

from dataclasses import dataclass, field
from typing import Optional


# ----------------------------------------------------------------------------------------------
# 1. Check dependencies
# ----------------------------------------------------------------------------------------------
def _require(package:str, install_cmd:str) -> None:

    import importlib.util

    if importlib.util.find_spec(package) is None:
        print(f"\n[ERROR] Required package {package} not found."
              f"\n      Install it with: {install_cmd}\n")
        sys.exit(1)

_require('vaderSentiment', 'pip install vaderSentiment')

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ----------------------------------------------------------------------------------------------
# 2. Aspect taxonomy
# ----------------------------------------------------------------------------------------------

# Each key is the canonical aspect name shown in the output.
# Each value is a list of surface form keywords (singular, plural, synonyms).
# Matching is case-sensitive and whole-word only (regext \b boundaries)
ASPECT_TAXONOMY: dict[str, list[str]] = {
    "battery life":   ["battery life", "battery", "charge", "charging"],
    "screen":         ["screen", "display", "monitor", "resolution", "brightness"],
    "camera":         ["camera", "cameras", "photo", "photos", "lens", "zoom"],
    "performance":    ["performance", "speed", "processor", "lag", "fast", "slow", "snappy"],
    "price":          ["price", "cost", "expensive", "cheap", "affordable", "value"],
    "service":        ["service", "services", "support", "staff", "customer service"],
    "build quality":  ["build quality", "build", "design", "materials", "durability", "sturdy"],
    "storage":        ["storage", "memory", "space", "capacity"],
    "audio":          ["audio", "sound", "speaker", "speakers", "headphone", "headphones", "bass"],
}

# VADER compound-score thresholds (industry standard values)
POSITIVE_THRESHOLD =  0.05
NEGATIVE_THRESHOLD = -0.05

# ----------------------------------------------------------------------------------------------
# 3. Data classes
# ----------------------------------------------------------------------------------------------
@dataclass
class AspectResult:
    """Sentiment prediction for one aspect within a sentence."""
    aspect:     str
    sentiment:  str          # "POSITIVE" | "NEUTRAL" | "NEGATIVE"
    score:      float        # VADER compound score in [-1, +1]
    confidence: str          # "HIGH" | "MEDIUM" | "LOW"
    source:     str          # the clause that the aspect was found in


@dataclass
class SentenceResult:
    """Full ABSA output for one input sentence."""
    text:           str
    aspects:        list[AspectResult] = field(default_factory=list)
    overall_label:  Optional[str] = None   # overall VADER sentiment
    overall_score:  Optional[float] = None