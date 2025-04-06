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

- `GoogleFactCheck` module aggregates data from (https://toolbox.google.com/factcheck/explorer/search/list:recent;hl=en)[Google Fact Check Tools] to get credibility scores
- `ClaimBuster` module aggregates data from (https://idir.uta.edu/claimbuster/api/)[ClaimBuster API] to get credibility scores

**Vector Database**

We provide a vectorized versions of two popular fact-checking databases: 
- (https://paperswithcode.com/dataset/liar)[LIAR]
- (https://fever.ai/dataset/adversarial.html)[FEVER v2.0]

See `scripts/vectorized.py` and `indices` for our vectorization script and resulting data storages 