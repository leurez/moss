import warc
import gzip

# Download the Common Crawl dataset using the AWS CLI
!aws s3 sync s3://commoncrawl/crawl-data/CC-MAIN-2021-39/ /data/commoncrawl/

# Extract text from the WARC files
with gzip.open('/data/commoncrawl/CC-MAIN-2021-39/segments/1632206446796.22/warc/CC-MAIN-20210920165830-20210920205830-00000.warc.gz', 'rb') as f:
    warc_file = warc.WARCFile(fileobj=f)
    for record in warc_file:
        if record['WARC-Type'] == 'response':
            content = record.payload.read()
            # Extract text from the HTML content using a library like BeautifulSoup
