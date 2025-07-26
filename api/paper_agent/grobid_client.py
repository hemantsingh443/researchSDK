import requests
import os
from typing import Optional

class GrobidClient:
    def __init__(self, grobid_url: Optional[str] = None):
        self.grobid_url = grobid_url or os.getenv("GROBID_URL", "http://localhost:8070")
        if not self.grobid_url.endswith("/api"):
            self.grobid_url = self.grobid_url.rstrip('/') + "/api"

    def process_pdf(self, pdf_path: str) -> Optional[str]:
        """
        Sends a PDF to Grobid and returns the TEI XML as a string.
        """
        if not os.path.exists(pdf_path):
            print(f"Error: PDF file not found at {pdf_path}")
            return None

        process_url = f"{self.grobid_url}/processFulltextDocument"
        print(f"Sending '{pdf_path}' to Grobid at '{process_url}'...")

        try:
            with open(pdf_path, 'rb') as f:
                files = {'input': f}
                data = {'consolidateHeader': '1'}
                response = requests.post(process_url, files=files, data=data, timeout=120)

            if response.status_code == 200:
                print("Grobid processing successful.")
                return response.text
            else:
                print(f"Error processing with Grobid. Status: {response.status_code}, Response: {response.text}")
                return None
        except requests.exceptions.RequestException as e:
            print(f"Grobid request failed: {e}")
            return None