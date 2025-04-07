import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple
import pandas as pd

class VectorSearcher:
    def __init__(self, index_path: str, dataframe_path: str, config: Dict):
        """
        Initialize the vector searcher with FAISS index and associated data
        
        Args:
            index_path: Path to the FAISS index file
            dataframe_path: Path to the associated DataFrame (CSV)
            config: Configuration dictionary containing:
                   - text_column: Name of the text column
                   - label_column: Name of the label column
                   - label_mapping: (Optional) Dictionary for label conversion
        """
        self.index = faiss.read_index(index_path)
        self.df = pd.read_csv(dataframe_path)
        self.config = config
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
    def vectorize_input(self, text: str) -> np.ndarray:
        """Convert text input to embedding vector"""
        return self.model.encode([text], convert_to_numpy=True)
    
    def search(self, query_vector: np.ndarray, k: int = 3) -> List[Dict]:
        """
        Search the index for similar vectors
        
        Args:
            query_vector: Vectorized input query
            k: Number of results to return
            
        Returns:
            List of result dictionaries with text, label, and distance
        """
        distances, indices = self.index.search(query_vector, k)
        results = []
        
        for i, idx in enumerate(indices[0]):
            if idx == -1:  # Handle cases where fewer than k results exist
                continue
                
            match = self.df.iloc[idx]
            result = {
                'text': match[self.config['text_column']],
                'label': self._format_label(match[self.config['label_column']]),
                'distance': float(distances[0][i])
            }
            results.append(result)
            
        return results
    
    def _format_label(self, label_value) -> str:
        """Convert label value to human-readable format"""
        if 'label_mapping' in self.config:
            return self.config['label_mapping'].get(str(label_value), str(label_value))
        return str(label_value)

# Configuration for each dataset
FEVER_CONFIG = {
    'text_column': 'claim',
    'label_column': 'label'
}

LIAR_CONFIG = {
    'text_column': 'statement',
    'label_column': 'label',
    'label_mapping': {
        '0': 'pants-fire',
        '1': 'false',
        '2': 'barely true',
        '3': 'half-true',
        '4': 'mostly true',
        '5': 'true'
    }
}

# Initialize searchers (assuming you've saved your DataFrames as CSVs)
fever_searcher = VectorSearcher(
    index_path='indices/fever_index.faiss',
    dataframe_path='data/fever_data.csv',
    config=FEVER_CONFIG
)

liar_searcher = VectorSearcher(
    index_path='indices/liar_index.faiss',
    dataframe_path='data/liar_data.csv',
    config=LIAR_CONFIG
)

def check_claim(text: str) -> Dict:
    """
    Check a claim against both datasets and return combined results
    
    Args:
        text: Input text to verify
        
    Returns:
        Dictionary with results from both datasets
    """
    vector = fever_searcher.vectorize_input(text)
    
    return {
        'fever_results': fever_searcher.search(vector),
        'liar_results': liar_searcher.search(vector)
    }

# Example usage
if __name__ == "__main__":
    while True:
        user_input = input("\nEnter a claim to verify (or 'quit' to exit): ")
        if user_input.lower() == 'quit':
            break
            
        results = check_claim(user_input)
        
        print("\nFEVER Results:")
        for res in results['fever_results']:
            print(f"- {res['text']} (Label: {res['label']}, Distance: {res['distance']:.2f})")
            
        print("\nLIAR Results:")
        for res in results['liar_results']:
            print(f"- {res['text']} (Label: {res['label']}, Distance: {res['distance']:.2f})")