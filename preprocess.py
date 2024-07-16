import asyncio
import aiohttp
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec

# Pinecone setup
api_key = "516cbfbd-5e65-4050-b31e-5ff69efa9f23"
pc = Pinecone(api_key=api_key, environment="us-east-1")
index_name = "example-index-2"

index = pc.Index(index_name)

# Setup model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

async def fetch_data_from_url(url):
    try:
        async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=False)) as session:
            async with session.get(url) as response:
                content = await response.text()
                soup = BeautifulSoup(content, "html.parser")
                return soup.get_text(), soup.title.string if soup.title else None
    except Exception as e:
        print(f"Error fetching data from {url}: {e}")
        return None, None

def embed_text(text):
    try:
        embedding = model.encode(text)
        return embedding
    except Exception as e:
        print(f"Error embedding text: {e}")
        return None

async def upload_to_pinecone(data):
    embeddings = []

    for url, (text, _) in data.items():
        embedding = embed_text(text)
        if embedding is not None:
            embeddings.append({
                "id": url,
                "values": embedding.tolist(),
                # "metadata": {"text": text}
            })
            print(f"Embedding for {url}")

    try:
        index.upsert(vectors=embeddings)
        print("Upload successful.")
    except Exception as e:
        print(f"Error upserting vectors to Pinecone index: {e}")

async def process_urls(urls):
    data = {}
    
    # Create a list of tasks for concurrent fetching
    tasks = [fetch_data_from_url(url) for url in urls]
    
    # Run all tasks concurrently and wait for them to complete
    results = await asyncio.gather(*tasks)

    # Store the results in the data dictionary
    for url, (text, title) in zip(urls, results):
        if text:
            data[url] = (text, title)

    # Upload the data to Pinecone
    await upload_to_pinecone(data)

if __name__ == "__main__":
    urls = [
        "https://en.wikipedia.org/wiki/Artificial_intelligence",
        "https://en.wikipedia.org/wiki/Machine_learning",
        "https://www.bbc.com/news",
        "https://www.nasa.gov/",
        "https://pytorch.org/",
        "https://openai.com/",
        "https://hbr.org/",
        "https://logbinary.com/",
        "https://www.discovery.com/",
        "https://www.nytimes.com/",
        "https://www.nationalgeographic.com/",
        "https://www.nature.com/",
        "https://www.isro.gov.in/",
        "https://timesofindia.indiatimes.com/",
        "https://www.wsj.com/",
        "https://ai.googleblog.com/",
        "https://www.cnbc.com/",
        "https://www.un.org/",
        "https://www.nih.gov/",
        "https://www.sciencemag.org/",
        "https://www.lonelyplanet.com/",
        "https://www.gamespot.com/",
        "https://www.ign.com/",
        "https://www.cnet.com/",
        "https://logbinary.com/"
    ]

    asyncio.run(process_urls(urls))
