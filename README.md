Semantic Search on Twitter API Documentation
Points: 30
Goal: Build a semantic search engine over the full Twitter API Postman documentation.**
Invocation Requirement
Your solution must allow command-line querying:
python semantic_search.py --query "How do I fetch tweets with expansions?"
This should print the top-k most relevant documentation chunks.
Implementation Requirements
Use the GitHub repo: https://github.com/xdevplatform/postman-twitter-api
Chunk documentation intelligently.
Embed chunks (model of your choice).
Build a vector index (FAISS/Chroma/custom).
Implement top‑k semantic retrieval.
Output: JSON printed to stdout with ranked chunks.
