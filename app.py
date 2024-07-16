from flask import Flask, request, jsonify, render_template
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import json
import re

app = Flask(__name__)

# Pinecone setup
api_key = "516cbfbd-5e65-4050-b31e-5ff69efa9f23"
pc = Pinecone(api_key=api_key, environment="us-east-1")
index_name = "example-index"
index = pc.Index(index_name)

# Setup model
model = SentenceTransformer('all-mpnet-base-v2')
# Load fetched data
with open("fetched_data.json", "r") as f:
    fetched_data = json.load(f)

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

from flask import Flask, request, render_template_string
import re
import json

@app.route('/query', methods=['POST'])
def query():
    query_text = request.json.get('query_text', '')
    query_vector = embed_text(query_text)

    output_lines = [f"Query results for '{query_text}':", ""]

    if query_vector is not None:
        try:
            query_results = index.query(vector=query_vector.tolist(), top_k=10, include_metadata=True)

            if 'matches' in query_results and query_results['matches']:
                max_score = max(result['score'] for result in query_results['matches'])
                for result in query_results["matches"]:
                    url = result['id']
                    text = fetched_data.get(url, "")
                    score = result['score']
                    percentage = (score / max_score) * 100 if max_score != 0 else 0

                    output_lines.append(f"URL: {url}")
                    output_lines.append('-' * 50)

                    if text:
                        matching_paragraphs = []
                        for paragraph in text.split("\n\n"):
                            if query_text.lower() in paragraph.lower():
                                # Bold the matching word using regex
                                highlighted_paragraph = re.sub(
                                    f"({re.escape(query_text)})",
                                    r"<strong>\1</strong>",
                                    paragraph,
                                    flags=re.IGNORECASE
                                )
                                matching_paragraphs.append(highlighted_paragraph)

                        if matching_paragraphs:
                            output_lines.append("Matching paragraphs:")
                            for match in matching_paragraphs:
                                output_lines.append(f"{match}")
                        else:
                            output_lines.append("No matching paragraphs found.")

                        output_lines.append(f"Score: {score:.2f}, Percentage: {percentage:.2f}%")
                    else:
                        output_lines.append(f"Error fetching content from {url}")

                    output_lines.append('-' * 50)

            else:
                output_lines.append(f"No matches found for '{query_text}'.")
        except Exception as e:
            output_lines.append(f"Error querying Pinecone index: {e}")
    else:
        output_lines.append("Error embedding the query text.")

    return jsonify({"message": "\n".join(output_lines)})


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
