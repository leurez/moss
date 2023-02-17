import requests
import os

# Download the WebText dataset
url = 'https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-raw-v1.zip'
response = requests.get(url)

# Save the dataset to a file
with open('wikitext-2-raw-v1.zip', 'wb') as f:
    f.write(response.content)

# Unzip the file
os.system('unzip wikitext-2-raw-v1.zip')

# Load the text files into memory and concatenate them into a single string
text = ''
with open('wikitext-2-raw/wiki.test.raw', 'r') as f:
    text += f.read()
with open('wikitext-2-raw/wiki.valid.raw', 'r') as f:
    text += f.read()
with open('wikitext-2-raw/wiki.train.raw', 'r') as f:
    text += f.read()

# Write the concatenated text to a file
with open('webtext.txt', 'w') as f:
    f.write(text)
