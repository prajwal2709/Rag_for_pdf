from pymupdf import open
import re
from typing import List
from pathlib import Path

def paragraph_extract(pdf_path:str)->List[str]:
    """
    extract paragrpahs from pdf file"""

    doc=open(pdf_path)
    paragraphs=[]

    for page in doc:
        blocks=page.get_text("blocks")
        blocks=sorted(blocks,key=lambda b:(b[1],b[0])) 

        for block in blocks:
            text=block[4]

            text=re.sub(r'\s+',' ',text).strip()
            if text:
                paragraphs.append(text)
    return paragraphs

print(len(paragraph_extract("pdfs/penguins_ACL.pdf ")))
