import requests, os
from typing import Optional, List
from functools import cache

class FactCheckAPI:
    """Base class for fact-checking API integrations"""
    @staticmethod
    def _make_request(url: str, headers: Optional[dict] = None, params: Optional[dict] = None) -> dict:
        """Request handler with error checking"""
        try:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"API request failed: {str(e)}")
            return {}
        

class GoogleFactCheck(FactCheckAPI):
    BASE_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
    
    def __init__(self):
        self.api_key = os.getenv("GOOGLE_FACTCHECK_API_KEY")
        if not self.api_key:
            raise ValueError("Google FactCheck API key not configured")

    def get_results(self, text: str) -> List[str]:
        """Get fact-check ratings for a given text"""
        params = {
            "query": text,
            "key": self.api_key
        }
        
        data = self._make_request(self.BASE_URL, params=params)
        return [
            review.get('textualRating', 'N/A')
            for claim in data.get("claims", [])
            for review in claim.get("claimReview", [])
        ]
    

class ClaimBuster(FactCheckAPI):
    BASE_URL = "https://idir.uta.edu/claimbuster/api/v2/score/text"
    
    def __init__(self):
        self.api_key = os.getenv("CLAIMBUSTER_API_KEY")
        if not self.api_key:
            raise ValueError("ClaimBuster API key not configured")
        self.headers = {"x-api-key": self.api_key}

    @cache
    def get_score(self, text: str) -> float:
        """Get claim score for a given text (with caching)"""
        url = f"{self.BASE_URL}/{text}"
        data = self._make_request(url, headers=self.headers)
        return data.get('results', [{}])[0].get('score', 0.0)