import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocessing_text(text):

    """
    Following preprocessing steps to be done
    1. Lowercasing
    2. Remvoing Special Characters
    3. Tokenization
    4. Stopword removal
    5. Lemmatization
    """

    # 1. Lowercasing
    text = text.lower()

    # 2. Removing Special Characters
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)

    # 3. Tokenization
    tokens = word_tokenize(text)

    # 4. Remove Stopwords
    Tokens = [word for word in tokens if word not in stop_words]

    # 5. Lemmatization
    tokens = [lemmatizer.lemmatize(word) for word in Tokens]

    # Convert back to string

    cleaned_text = " ".join(tokens)


    return cleaned_text