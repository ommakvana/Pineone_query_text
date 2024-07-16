from flask import Flask, request, jsonify,render_template
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone

app = Flask(__name__)

# Pinecone setup
api_key = "516cbfbd-5e65-4050-b31e-5ff69efa9f23"
pc = Pinecone(api_key=api_key, environment="us-east-1")
index_name = "example-index-2"

index = pc.Index(index_name)

# Setup model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def embed_text(text):
    try:
        embedding = model.encode(text)
        return embedding
    except Exception as e:
        print(f"Error embedding text: {e}")
        return None

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

                output += f"URL: {url}\n"
                output += '-' * 50 + '\n'
                output += f"Score: {score:.2f}, Percentage: {percentage:.2f}%\n\n"
                output += '-' * 50 + '\n'

        else:
            output += f"No matches found for '{query_text}'."
    else:
        output += "Error embedding the query text."

    return jsonify({"message": output})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
