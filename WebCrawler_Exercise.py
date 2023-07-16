from bs4 import BeautifulSoup
import requests
import sqlite3
import string


class WebCrawler:
    def __init__(self, seed_urls):
        self.queue = list(seed_urls)
        self.connection = sqlite3.connect('index.sqlite')  # Connect to SQLite database
        self.cursor = self.connection.cursor()
        self.create_index_table()

    # ---------------------------------------
    
    def create_index_table(self):
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS IndexTable (
                                term TEXT,
                                url TEXT,
                                frequency INTEGER
                            )''')
        self.connection.commit()

    # ---------------------------------------

    def fetch_page(self, url):
        response = requests.get(url)
        if response.status_code == 200:
            print('Fetched successfully for', url, 'with status code:', response.status_code)
            return response.content
        else:
            print('The URL', url, 'refused the connection. Status code:', response.status_code)
            return None
        
    # ---------------------------------------
    
    def parse_page(self, html, url):
        soup = BeautifulSoup(html, 'html.parser')

        text_content = soup.get_text()
        raw_words = text_content.split()

        terms = self.normalize_words(raw_words)

        # Create a set to store unique terms for the URLs
        unique_terms = set()
    
        # Create a dictionary to store term frequencies
        term_info = {}


        # Count term frequencies
        for term in terms:
            term_info[term] = term_info.get(term, 0) + 1

        # Insert the indexed terms, URLs, and frequencies into the database
        for term, frequency in term_info.items():
            if term not in unique_terms:
                self.cursor.execute("INSERT INTO IndexTable VALUES (?, ?, ?)", (term, url, frequency))
                unique_terms.add(term)
        self.connection.commit()

    # ---------------------------------------

    def normalize_words(self, words):
        # list of stop words
        stop_words = ["the", "and", "in", "to", "is", "it", "of"]
        
        normalized_words = []

        for word in words:
            # Remove punctuation
            word = word.translate(str.maketrans('', '', string.punctuation))

            # Remove leading/trailing whitespaces
            word = word.strip()

            # Convert to lowercase
            word = word.lower()

            # Check if word is a stop word
            if word and word not in stop_words:
                normalized_words.append(word)

        return normalized_words

    # ---------------------------------------

    def crawl(self):
        print('Start Crawling Operation:')
        while self.queue:
            url = self.queue.pop(0)
            html = self.fetch_page(url)
            if html is not None:
                self.parse_page(html, url)
        print('\nCrawling Successfully Finished.')

    # ---------------------------------------

    def search(self, query):
        # Convert query to lowercase and split into individual terms
        query_terms = query.lower().split()  
        doc_urls = set()

        for term in query_terms:
            self.cursor.execute("SELECT url FROM IndexTable WHERE term=?", (term,))
            result = self.cursor.fetchall()
            if result:
                doc_urls.update([row[0] for row in result])

        return [doc_url for doc_url in doc_urls]

    # ---------------------------------------

    def close_connection(self):
        self.cursor.close()
        self.connection.close()
        
# =================================================================
        
def print_line():
    print();print('-'*80);print('-'*80);print()

# =================================================================

if __name__ == '__main__':
    seed_urls = [
        'https://www.basu.ac.ir',
        'https://www.mediawiki.org',
        'https://www.wikipedia.org',
        'https://www.duckduckgo.com',
        'https://www.bing.com',
        'https://www.msn.com',
        'https://www.downloadha.com',
        'https://www.google.com',
        'https://www.yahoo.com',
        'https://bitpin.ir/',
        'https://linuxcommand.org',
        'https://www.wsj.com/',
        'https://www.pnas.org',
        'https://www.downloadly.ir',
        'https://www.uptvs.com',
        'https://www.theguardian.com/international',
        'https://www.goal.com/en',
        'https://www.bbc.com',
        'https://www.bbc.com/sport/football',
    ]

    crawler = WebCrawler(seed_urls)
    crawler.crawl()

    print_line()

    # Searching for terms in the indexed information
    try:
        while True:
            search_terms = input('Enter your terms to search (separated by space): ')
            search_results = crawler.search(search_terms)
            print('\nSearch results for', search_terms, 'are:')
            if len(search_results) == 0:
                print('Nothing Found :/')
            else:
                for result in search_results:
                    print(result)
            print_line()
    except KeyboardInterrupt:
        pass

    # Close the database connection when done
    crawler.close_connection()
