@app.route('/query', methods=['POST'])
def query():
    query_text = request.json.get('query_text', '')
    query_vector = embed_text(query_text)

    output = f"Query results for '{query_text}':\n\n"

    if query_vector is not None:
        try:
            query_results = index.query(vector=query_vector.tolist(), top_k=10, include_metadata=True)

            if 'matches' in query_results and query_results['matches']:
                max_score = max(result['score'] for result in query_results['matches'])
                urls = [result['id'] for result in query_results['matches']]

                # Fetch data concurrently
                fetched_data = asyncio.run(fetch_multiple_urls(urls))

                for result, (text, title) in zip(query_results["matches"], fetched_data):
                    score = result['score']
                    percentage = (score / max_score) * 100 if max_score != 0 else 0
                    url = result['id']

                    output += f"URL: {url}\n"
                    output += '-' * 50 + '\n'

                    if text:
                        matching_words = []
                        for paragraph in text.split("\n\n"):
                            if query_text.lower() in paragraph.lower():
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
        except Exception as e:
            output += f"Error querying Pinecone index: {e}"
    else:
        output += "Error embedding the query text."

    return jsonify({"message": output})