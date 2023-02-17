import os
import json
import re
import nltk

nltk.download('punkt')

# set paths to SPGC files
spgc_path = "/path/to/SPGC"
metadata_file = os.path.join(spgc_path, "metadata.json")
content_dir = os.path.join(spgc_path, "content")
output_file = os.path.join(spgc_path, "spgc.txt")

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

# read in metadata file and clean text
with open(metadata_file, "r") as f, open(output_file, "w") as out_file:
    metadata = json.load(f)
    for document in metadata:
        # read in content of document
        with open(os.path.join(content_dir, document["filename"]), "r") as content_file:
            content = content_file.read()
            # clean text of document
            cleaned_content = clean_text(content)
            # write cleaned text to output file
            out_file.write(cleaned_content + "\n")
