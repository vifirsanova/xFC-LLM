import pandas as pd
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import faiss
import os

def load_and_process_data(dataset_name, dataset_version, split, text_column, index_output_path):
    """
    Load dataset, process it, generate embeddings, and create FAISS index.
    
    Args:
        dataset_name: Name or path of the dataset
        split: Dataset split to use (e.g., 'train', 'validation')
        text_column: Column name containing the text to embed
        index_output_path: Path to save the FAISS index
        
    Returns:
        Tuple of (DataFrame, embeddings, FAISS index)
    """
    # Load dataset
    dataset = load_dataset(dataset_name, dataset_version, trust_remote_code=True if dataset_name == "liar" else None)
    df = pd.DataFrame(dataset[split])
    
    # Generate embeddings
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(df[text_column].tolist(), convert_to_numpy=True)
    
    # Create and populate FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    
    # Save index
    os.makedirs(os.path.dirname(index_output_path), exist_ok=True)
    faiss.write_index(index, index_output_path)
    
    return df, embeddings, index

def load_index(index_path, embeddings_shape):
    """
    Load FAISS index from disk.
    
    Args:
        index_path: Path to the saved FAISS index
        embeddings_shape: Expected shape of embeddings (for validation)
        
    Returns:
        FAISS index
    """
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Index file not found at {index_path}")
    
    index = faiss.read_index(index_path)
    if index.d != embeddings_shape[1]:
        raise ValueError("Embedding dimension mismatch between loaded index and expected shape")
    
    return index

# Process and save LIAR dataset
df_liar, embeddings_liar, index_liar = load_and_process_data(
    dataset_name="liar",
    dataset_version=None,
    split="train",
    text_column="statement",
    index_output_path="indices/liar_index.faiss"
)

# Process and save FEVER dataset
df_fever, fever_embeddings, index_fever = load_and_process_data(
    dataset_name='fever/fever',
    dataset_version='v2.0',
    split="validation",
    text_column="claim",
    index_output_path="indices/fever_index.faiss"
)