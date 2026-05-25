from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama

# STEP 1: Read notes
with open("notes.txt", "r", encoding="utf-8") as file:
    text = file.read()

print("\nNOTES LOADED\n")

# STEP 2: Split text into chunks
splitter = CharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)

chunks = splitter.split_text(text)

print(f"Created {len(chunks)} chunks\n")

# STEP 3: Convert chunks into embeddings
embeddings = OllamaEmbeddings(model="llama3")

# STEP 4: Store embeddings in vector DB
db = Chroma.from_texts(chunks, embeddings)

print("Vector database ready\n")

# STEP 5: Ask question
query = input("Ask your question: ")

# STEP 6: Search similar chunks
docs = db.similarity_search(query, k=2)

print("\nRetrieved Chunks:\n")

for doc in docs:
    print(doc.page_content)
    print("------")

# STEP 7: Combine context
context = "\n".join([doc.page_content for doc in docs])

# STEP 8: Load LLM
llm = Ollama(model="llama3")

# STEP 9: Create prompt
prompt = f"""
Answer ONLY using this context:

{context}

Question: {query}
"""

# STEP 10: Generate answer
response = llm.invoke(prompt)

print("\nANSWER:\n")
print(response)