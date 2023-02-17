import json

# Read the downloaded Wikipedia dump file
with open('enwiki-latest-pages-articles.xml.bz2', 'rb') as f:
    wiki_dump = f.read()

# Parse the dump using the Wikipedia Extractor library
from wikiextractor import WikiExtractor

wiki_extractor = WikiExtractor()
wiki_text = wiki_extractor.extract(wiki_dump)

# Write the extracted text to a file
with open('wikipedia_text.txt', 'w') as f:
    for page in wiki_text:
        f.write(json.dumps(page) + '\n')
