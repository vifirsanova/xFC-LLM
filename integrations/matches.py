from typing import Dict, Union
import pandas as pd
import faiss

class FactChecker:
    def __init__(self, model, index: faiss.Index, dataframe: pd.DataFrame, config: Dict):
        """
        Initialize FactChecker with model, FAISS index, dataframe, and configuration.
        
        Args:
            model: SentenceTransformer model for embeddings
            index: FAISS index for similarity search
            dataframe: DataFrame containing claims/statements and labels
            config: Dictionary with configuration including:
                   - text_column: name of text column
                   - label_column: name of label column
                   - label_mapping: (optional) dict mapping label values to strings
        """
        self.model = model
        self.index = index
        self.dataframe = dataframe
        self.config = config
        
    def get_results(self, text: str, k: int = 1) -> str:
        """
        Find similar claims/statements and return formatted results.
        
        Args:
            text: Input text to check
            k: Number of similar results to return
            
        Returns:
            Formatted string with results
        """
        # Generate embedding for input text
        embedding = self.model.encode([text], convert_to_numpy=True)
        
        # Search for similar items
        distances, indices = self.index.search(embedding, k=k)
        
        # Get matching rows from dataframe
        matches = self.dataframe.iloc[indices[0]]
        
        # Format results
        results = []
        for idx, match in matches.iterrows():
            text_content = match[self.config['text_column']]
            label = self._format_label(match[self.config['label_column']])
            
            results.append(
                f"This text is similar to <{text_content}>, {label}"
            )
        
        return "\n".join(results)
    
    def _format_label(self, label_value: Union[int, str]) -> str:
        """
        Format label value according to configuration.
        
        Args:
            label_value: Raw label value from dataframe
            
        Returns:
            Formatted label string
        """
        if 'label_mapping' in self.config:
            return self.config['label_mapping'].get(str(label_value), f"labeled: {label_value}")
        return f"the credibility label is: {label_value}"


# Configuration for each dataset
FEVER_CONFIG = {
    'text_column': 'claim',
    'label_column': 'label',
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


# Initialize fact checkers
fever_checker = FactChecker(
    model=model,
    index=index_fever,
    dataframe=df_fever,
    config=FEVER_CONFIG
)

liar_checker = FactChecker(
    model=model,
    index=index_liar,
    dataframe=df_liar,
    config=LIAR_CONFIG
)

# Example usage:
fever_result = fever_checker.get_results("Your claim to check")
liar_result = liar_checker.get_results("Your statement to check")