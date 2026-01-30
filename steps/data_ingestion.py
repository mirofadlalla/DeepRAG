from zenml import step
import os 
from typing import  List , Dict
from pathlib import Path 
import hashlib 
import logging 

import fitz

'''
document-level parsing.
الهدف

نطلع وحدات نص نظيفة + metadata مضبوط
بحيث أي chunk بعد كده يعرف هو جاي منين بالظبط.

هنطلع Pages / Blocks بالشكل ده:
{
  "text": "...",
  "metadata": {
      "source": "doc_name",
      "page": 3,
      "language": "ar",
      "doc_type": "article"
  }
}

'''


# we will work on en + ar only so  we have to detect the lang

def detect_lang(text : str ) -> str :

    ar_chars = sum(1 for char in text if '\u0600' <= char <= '\u06FF')
    return "ar" if ar_chars > 10 else "en"

class DocumentLoader:
    def __init__(self , data_path : Path):
        self.path = Path(data_path)

    def _load_txt(self , data_path : Path) -> List[Dict]:
        '''' if text is a type of txt we use this method to extract it'''
        try :
            logging.info("extracting the text from txt file . ")
            text = data_path.read_text(encoding="utf-8") 
            return [
                {
                    "text" : text , 
                    "metadata" : {
                        "source" : data_path.name , 
                        "page" : 1 , 
                        "language" : detect_lang(text) , 
                        "doc_type": "txt"
                    }
                }
            ]
        
        except Exception as e :
            logging.error("Couldin't extract the text .....")
            raise e
        

    def _load_pdf(self , data_path : Path) -> List[Dict]:
        '''' if text is a type of pdf we use this method to extract it'''

        pages = []
        try :
            logging.info(f"extracting the text from {data_path.name}.pdf file . ")
            doc = fitz.open(data_path)

            for page_number , page in enumerate(doc):
                text = page.get_text().strip()
                if not text :
                    continue 


                pages.append({
                    "text": text,
                    "metadata": {
                        "source": data_path.name,
                        "page": page_number + 1,
                        "language": detect_lang(text),
                        "doc_type": "pdf"
                    }
                })
                logging.info(f"text extracted sussfully from {data_path.name}.pdf file ")
        except Exception as e :
            logging.error(f"Couldin't open and extracting the {data_path.name}.pdf file .....")
            raise e
        

        return pages
    

@step(enable_cache=False)
def load(data_path : Path) -> List[Dict]:
        ''' a general method of extracting documents '''
        loader = DocumentLoader(data_path)
        docs = []

        for file in data_path.iterdir():
            if file.suffix.lower() == '.txt' :
                docs.extend(loader._load_txt(file))
            elif file.suffix.lower() == ".pdf":
                docs.extend(loader._load_pdf(file))

        if len(docs) > 1 :
            logging.info("All Text Has been Extracted Sussfully .....")
        return docs

