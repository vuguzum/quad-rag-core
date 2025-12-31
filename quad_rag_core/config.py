# quadant_rag_core/config.py

# Number of words in a chunk
CHUNK_SIZE_WORDS = 150
CHUNK_OVERLAP_RATIO = 0.15  # 15% overlap
# Similarity threshold (cosine similarity for nomic-embed-text ~ [-1, 1], but usually [0.2, 0.95])
SEARCH_SCORE_THRESHOLD = 0.150
# Preview length (characters)
CHUNK_CHARACTERS_PREVIEW = 100
# Threshold for reranking (cross-encoder score)
RERANK_SCORE_THRESHOLD = 0.35

TEXT_FILE_EXTENSIONS = {
    # Programming languages
    '.c', '.cpp', '.cs', '.csproj', '.go', '.h', '.hpp', '.java', '.js', '.php',
    '.py', '.rb', '.rs', '.sln', '.ts',

    # Scripts and configs
    '.bat', '.cfg', '.ini', '.sh', '.toml', '.yaml', '.yml',

    # Markup and web
    '.txt', '.css', '.html', '.ipynb', '.json', '.log', '.md', '.xml',
}