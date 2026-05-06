
from pathlib import Path
from llama_index.core import Document
from llama_index.readers.file import PDFReader, DocxReader, PptxReader
from llama_index.readers.file import FlatReader
from llama_index.core import Settings


import logging
SUPPORTED = {".pdf", ".txt", ".md", ".docx", ".pptx"}
logger=logging.getLogger(__name__)



def extract_file(file):
    
        path= file if isinstance (file,str) else file.name
        p=Path(path)
        ext=p.suffix.lower()
        if ext not in SUPPORTED:
              raise ValueError(f"Unsupported file type {ext}. Supported: {sorted(SUPPORTED)}")
        
        try:
               if ext== ".pdf":
                      Doc = PDFReader().load_data(path)
                      return Doc  
               elif ext==".docx":
                      Doc=DocxReader().load_data(path)
                      return Doc  
               elif ext == ".pptx":
                      Doc=PptxReader().load_data(path)
                      return Doc  
               elif ext in {".txt",".md"}:
                      Doc=FlatReader().load_data(path)
                      return Doc         
        except Exception as e:
           logger.error (f"error in {e}")
           return {}

        