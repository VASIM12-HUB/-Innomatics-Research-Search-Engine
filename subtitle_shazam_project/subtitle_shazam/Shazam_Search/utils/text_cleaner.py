import re

def clean_text(text):
    text = re.sub(r"\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}", "", text)
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"[^\w\s.,!?]", "", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text