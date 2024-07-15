import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from flask import Flask, request, jsonify, render_template
from pinecone import Pinecone, ServerlessSpec

app = Flask(__name__)

# Pinecone setup
api_key = "516cbfbd-5e65-4050-b31e-5ff69efa9f23"
pc = Pinecone(api_key=api_key, environment="us-east-1")
index_name = "example-index-2"

if index_name not in pc.list_indexes().names():
    pc.create_index(name=index_name, dimension=384, metric="cosine", spec=ServerlessSpec(cloud="aws", region="us-east-1"))

index = pc.Index(index_name)

# Setup model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def fetch_data_from_url(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, "html.parser")
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

def upload_to_pinecone(data):
    embeddings = []

    for url, (text, _) in data.items():
        embedding = embed_text(text)
        if embedding is not None:
            embeddings.append({
                "id": url,
                "values": embedding.tolist(),
            })
            print(f"Embedding for {url}")

    try:
        index.upsert(vectors=embeddings)
        print("Upload successful.")
    except Exception as e:
        print(f"Error upserting vectors to Pinecone index: {e}")

def process_urls(urls):
    data = {}
    for url in urls:
        text, title = fetch_data_from_url(url)
        if text:
            data[url] = (text, title)

    upload_to_pinecone(data)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def query():
    query_text = request.json.get('query_text', '')
    query_vector = embed_text(query_text)

    output = f"Query results for '{query_text}':\n\n"

    if query_vector is not None:
        query_results = index.query(vector=query_vector.tolist(), top_k=10, include_metadata=True)

        if 'matches' in query_results and query_results['matches']:
            max_score = max(result['score'] for result in query_results['matches'])

            for result in query_results["matches"]:
                score = result['score']
                percentage = (score / max_score) * 100 if max_score != 0 else 0
                url = result['id']
                text, title = fetch_data_from_url(url)

                output += f"URL: {url}\n"
                output += '-' * 50 + '\n'

                if text:
                    matching_words = []

                    for paragraph in text.split("\n\n"):
                        if query_text.lower() in paragraph.lower():
                            start_idx = paragraph.lower().index(query_text.lower())
                            matching_words.append(query_text)
                            similar_words = [word for word in paragraph.split() if query_text.lower() in word.lower()]
                            matching_words.extend(similar_words)

                    if matching_words:
                        for match in set(matching_words):
                            output += f"Matching text: '{match}'\n"
                    else:
                        output += "No exact matches found.\n"

                    output += f"Score: {score:.2f}, Percentage: {percentage:.2f}%\n\n"
                else:
                    output += f"Error fetching content from {url}\n"

                output += '-' * 50 + '\n'

        else:
            output += f"No matches found for '{query_text}'."
    else:
        output += "Error embedding the query text."

    return jsonify({"message": output})


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
        "https://timesofindia.indiatimes.com/"
        "https://www.wsj.com/",
        "https://ai.googleblog.com/",
        "https://www.cnbc.com/",
        "https://www.un.org/",
        "https://www.nih.gov/",
        "https://www.sciencemag.org/",
        "https://www.theatlantic.com/",
        "https://www.bloomberg.com/",
        "https://www.lonelyplanet.com/",
        "https://www.gamespot.com/",
        "https://www.ign.com/",
        "https://www.cnet.com/",
        "https://logbinary.com/"
    ]

    process_urls(urls)

    app.run(host='0.0.0.0', port=5000, debug=True)
