import os
import json
import re
import nltk

nltk.download('punkt')

# set paths to BookCorpus files
bookcorpus_path = "/path/to/BookCorpus"
books_file = os.path.join(bookcorpus_path, "books_large_p1.jsonl")
output_file = os.path.join(bookcorpus_path, "bookcorpus.txt")

# function to clean text
def clean_text(text):
    # remove non-alphanumeric characters
    text = re.sub(r'\W+', ' ', text)
    # convert to lowercase
    text = text.lower()
    # tokenize text into words
    words = nltk.word_tokenize(text)
    # join words back into text
    cleaned_text = " ".join(words)
    return cleaned_text

# read in BookCorpus JSONL files and clean text
with open(books_file, "r") as f, open(output_file, "w") as out_file:
    for line in f:
        # parse JSON from each line in file
        book = json.loads(line)
        # clean text of book
        cleaned_book = clean_text(book["text"])
        # write cleaned text to output file
        out_file.write(cleaned_book + "\n")
