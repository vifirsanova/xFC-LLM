from integrations.scores import GoogleFactCheck, ClaimBuster

from dotenv import load_dotenv
load_dotenv()  

# Example usage
if __name__ == "__main__":
    # Set API keys as environment variables before running
    google_checker = GoogleFactCheck()
    claimbuster = ClaimBuster()
    
    sample_text = "Climate change is a hoax"
    
    print("Google FactCheck Results:", google_checker.get_results(sample_text))
    print("ClaimBuster Score:", claimbuster.get_score(sample_text))


# Example of how to load the indices later
# index_liar_loaded = load_index("indices/liar_index.faiss", embeddings_liar.shape)
# index_fever_loaded = load_index("indices/fever_index.faiss", fever_embeddings.shape)