import requests, re
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import pandas as pd
import time 
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor
from threading import Thread

 
class Link:
    def __init__(self, url, text, content = None, date = None, source = None):
        """Constructor of Link

        Args:
            url (str): url of the link
            text (str): text of the link
            content (str): contents of the link. Defaults to None.
            date (str): date of the link post. Defaults to None.
            source (str): source of contents, name of site. Defaults to None.
        """
        self.url = url
        self.text = text
        self.content = content
        self.date = date
        self.source = source

class Crawler:
    """The crawler object is an instance of load of a crawler with set settings
    """
    def __init__(self, url_scrap, terms, size = 18, workers = 8):
        """Constructor of Crawler

        Args:
            url_scrap (list): user pass urls to crawl 
            terms (list): terms to look for additional urls
            size (int, optional): ammount of size of queue. Defaults to 18.
            workers (int, optional): number of threads to work in. Defaults to 8.
        """
        self.wordsToFind = terms
        self.foundLinks = []
        self.crawlQueue = Queue(size)
        self.workers = ThreadPoolExecutor(max_workers=workers)
        for url in url_scrap:
            urlParsed = urlparse(url)
            self.parametro_url = "{}://{}".format(urlParsed.scheme, urlParsed.netloc)
            link = Link(self.parametro_url, "Init param")
            self.crawlQueue.put(link)
        
    def __lookUp(self, html):
        soup = BeautifulSoup(html, 'html.parser')
        linksFound = []
        for word in self.wordsToFind:
            links = soup.find_all('a',text=re.compile("\w*" + word + "\w*", re.UNICODE))
            linksFound.extend(links) 
        for link in linksFound:            
            url = link['href']
            linkText = link.get_text().strip() # Remove spaces left and right      
            if url.startswith('/'): url = urljoin(self.parametro_url, url)            
            link = Link(url,linkText)
            self.crawlQueue.put(link)
        
    def __scrapWeb(self, url):
        try:
            res = requests.get(url, timeout=(3, 30))
            return res
        except requests.RequestException:
            return
        
    def __scrapeCallback(self, res):
        result = res.result()
        if result and result.status_code == 200:
            self.__lookUp(result.text)
                   
    def __runCC(self, limit = 0):
        done = False
        while not done:            
            try:
                if limit:
                    if len(self.foundLinks) - 1 == limit:
                        raise Empty()

                link = self.crawlQueue.get(timeout=5)         
                # url check
                if any(urlparse(x.url).path == urlparse(link.url).path for x in self.foundLinks): continue              
                self.foundLinks.append(link)
                worker = self.workers.submit(self.__scrapWeb, link.url)
                worker.add_done_callback(self.__scrapeCallback)
                time.sleep(0.5)                   
            except Empty:
                done = True                
                return 
            except Exception as e:
                print(e)
                continue

    def start(self):
        mainWorker = Thread(target=self.__runCC, args=(self, 3))
        mainWorker.start()