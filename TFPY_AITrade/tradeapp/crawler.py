import requests, re
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import pandas as pd
import time 
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor
from threading import Thread

 
class Link:
    def __init__(self, url, text):
        self.url = url
        self.text = text

class Crawler:
    def __init__(self, url_scrap, terms):
        if (isinstance(terms, str)):
            terms = [terms]
        elif (isinstance(terms, list)):
            if (isinstance(url_scrap, str)):
                url_scrap = [url_scrap]
            elif (isinstance(url_scrap, list)):
                self.wordsToFind = terms
                self.foundLinks = []
                self.crawlQueue = Queue(20)
                self.workers = ThreadPoolExecutor(max_workers=8)
                for url in url_scrap:
                    urlParsed = urlparse(url)
                    self.parametro_url = "{}://{}".format(urlParsed.scheme, urlParsed.netloc)
                    link = Link(self.parametro_url, "Init param")
                    self.crawlQueue.put(link)
            else:
                raise TypeError("url_scrap type not string or list")
        else:
            raise TypeError("terms type is not string or lsit")
        
    def lookUp(self, html):
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
        
    def scrapWeb(self, url):
        try:
            res = requests.get(url, timeout=(3, 30))
            return res
        except requests.RequestException:
            return
        
    def scrapeCallback(self, res):
        result = res.result()
        if result and result.status_code == 200:
            self.lookUp(result.text)
                   
    def runCC(this, self, limit = 0):
        done = False
        while not done:            
            try:
                if limit:
                    if len(this.foundLinks) - 1 == limit:
                        raise Empty()

                link = this.crawlQueue.get(timeout=5)         
                # url check
                if any(urlparse(x.url).path == urlparse(link.url).path for x in this.foundLinks): continue              
                this.foundLinks.append(link)
                
                worker = this.workers.submit(this.scrapWeb, link.url)
                worker.add_done_callback(this.scrapeCallback)
                time.sleep(0.5)    
                    
            except Empty:
                self.ui.lEstadoActual.setText('Busqueda finalizada')
                done = True                
                return 
            except Exception as e:
                print(e)
                continue