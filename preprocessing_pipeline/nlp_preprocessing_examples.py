"""
NLP Preprocessing Examples (Tokenization, Lemmatization, Stopwords, NER, Stemming)

This file demonstrates common NLP preprocessing steps using NLTK.
"""

# ------------------------------------------------------------------------------------------
# 0. Import required modules
# ------------------------------------------------------------------------------------------
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk import pos_tag, ne_chunk

# ------------------------------------------------------------------------------------------
# 1. Download required NLTK data
# ------------------------------------------------------------------------------------------
resources = [
    "punkt",
    "stopwords",
    "wordnet",
    "averaged_perceptron_tagger",
    "maxent_ne_chunker",
    "words"
]

for resource in resources:
    try:
        nltk.data.find(resource)
    except LookupError:
        nltk.download(resource)

# ------------------------------------------------------------------------------------------
# 2. Tokenization
# ------------------------------------------------------------------------------------------
text = "Natural language processing is a fascinating field of study."
words = word_tokenize(text)

print("\nTokenization:")
print(words)

# ------------------------------------------------------------------------------------------
# 3. Lemmatization (Normalization)
# ------------------------------------------------------------------------------------------
lemmatizer = WordNetLemmatizer()
normalized_words = [lemmatizer.lemmatize(word) for word in words]

print("\nLemmatization (Normalization):")
print(normalized_words)

# ------------------------------------------------------------------------------------------
# 4. Stopword Removal
# ------------------------------------------------------------------------------------------
stop_words = set(stopwords.words("english"))
filtered_words = [word for word in words if word.lower() not in stop_words]

print("\nAfter Stopword Removal:")
print(filtered_words)

# ------------------------------------------------------------------------------------------
# 5. Named Entity Recognition (NER)
# ------------------------------------------------------------------------------------------
pos_tags = pos_tag(words)
ner_result = ne_chunk(pos_tags)

print("\nNamed Entity Recognition:")
print(ner_result)

# ------------------------------------------------------------------------------------------
# 6. Custom Sequence Tokenization
# ------------------------------------------------------------------------------------------
def sequence_tokenize(text, delimiter='.'):
    sequences = text.split(delimiter)
    return [seq.strip() for seq in sequences if seq.strip()]

long_text = "NLP is interesting. It involves text processing. It is widely used."
sequences = sequence_tokenize(long_text)

print("\nSequence Tokenization:")
print(sequences)

# ------------------------------------------------------------------------------------------
# 7. Stemming
# ------------------------------------------------------------------------------------------
porter = PorterStemmer()
stemmed_words = [porter.stem(word) for word in words]

print("\nStemming:")
print(stemmed_words)