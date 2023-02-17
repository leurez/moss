import urllib.request
from bs4 import BeautifulSoup
from ebooklib import epub

# Download the e-book in EPUB format
url = 'http://www.gutenberg.org/ebooks/1342.epub.noimages'
urllib.request.urlretrieve(url, '1342.epub')

# Extract the plain text content from the EPUB file
book = epub.read_epub('1342.epub')
text = ''
for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
    text += BeautifulSoup(item.get_content(), 'html.parser').get_text()

# Clean the text
text = text.replace('\r', '')
text = text.replace('\n', ' ')
text = ' '.join(text.split())

# Tokenize the text using NLTK
from nltk.tokenize import word_tokenize
tokens = word_tokenize(text)

# Convert the tokens into numerical representations using TensorFlow
from tensorflow.keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(tokens)
sequences = tokenizer.texts_to_sequences(tokens)
