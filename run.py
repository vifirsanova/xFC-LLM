from integrations.scores import GoogleFactCheck, ClaimBuster
from integrations.matches import check
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

    results = check(sample_text)
        
    print("\nFEVER Results:")
    for res in results['fever_results']:
        print(f"- {res['text']} (Label: {res['label']}, Distance: {res['distance']:.2f})")
            
    print("\nLIAR Results:")
    for res in results['liar_results']:
        print(f"- {res['text']} (Label: {res['label']}, Distance: {res['distance']:.2f})")