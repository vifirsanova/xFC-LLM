# xFC-LLM
*An AI-powered fact-checking tool for professional journalism*

### Overview

**xFC-LLM** is an LLM-based framework designed to emulate expert-driven fact-checking processes through automated verification techniques. The tool integrates three core methodologies from manual fact-checking: 

- Cross-Checking
- Discourse Marker Analysis
- Task-Specific Guardrails for Fact-Checking

### System Design

![System Design](https://github.com/user-attachments/assets/c7ccaf62-ce7a-4f8a-9f31-fd202709bc5f)

**API Integration**

- `GoogleFactCheck` module aggregates data from [Google Fact Check Tools](https://toolbox.google.com/factcheck/explorer/search/list:recent;hl=en) to get credibility scores
- `ClaimBuster` module aggregates data from [ClaimBuster API](https://idir.uta.edu/claimbuster/api/) to get credibility scores

Example usage:
```python
from integrations.scores import GoogleFactCheck, ClaimBuster

google_checker = GoogleFactCheck()
claimbuster = ClaimBuster()
    
sample_text = "Climate change is a hoax"
    
print("Google FactCheck Results:", google_checker.get_results(sample_text))
print("ClaimBuster Score:", claimbuster.get_score(sample_text))
```

Sample model output: 
```
Google FactCheck Results: ['False', 'False', 'Exaggerates', 'Misleading', 'False', 'Not the Whole Story', 'Spins the Facts', 'Four Pinocchios', 'False', 'False']
ClaimBuster Score: 0.5001685622
```

**Vector Database**

We provide a vectorized versions of two popular fact-checking databases: 
- [LIAR](https://paperswithcode.com/dataset/liar)
- [FEVER v2.0](https://fever.ai/dataset/adversarial.html)

See `scripts/vectorized.py`, folders `indices` & `data` for our vectorization script and resulting data storages (faiss and parquet for vector similarity search)

**Vector Similarity Search**

Our script searches for the most similar samples from LIAR and FEVER and retrieves matches and their L2-distance scores

Example usage:
```python
from integrations.matches import check

results = check(sample_text)
        
print("\nFEVER Results:")
for res in results['fever_results']:
    print(f"- {res['text']} (Label: {res['label']}, Distance: {res['distance']:.2f})")
            
print("\nLIAR Results:")
for res in results['liar_results']:
    print(f"- {res['text']} (Label: {res['label']}, Distance: {res['distance']:.2f})")
```

Sample model output:
```
FEVER Results:
- Proponents of globalism tend to advocate for modification of economic policy (Label: SUPPORTS, Distance: 1.32)
- There exists a 2000 page novel called The Winds of Winter. (Label: NOT ENOUGH INFO, Distance: 1.43)
- There exists a vegan and an atheist called Ted Cruz. (Label: NOT ENOUGH INFO, Distance: 1.43)

LIAR Results:
- Global warming is a hoax. (Label: true, Distance: 0.24)
- Says Donald Trump says climate change is a hoax invented by the Chinese. (Label: barely true, Distance: 0.45)
- Only 3 percent of voters 18 to 34 dont believe that climate change is really happening. (Label: half-true, Distance: 0.62)
```