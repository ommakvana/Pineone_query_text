document.addEventListener("DOMContentLoaded", function() {
    document.getElementById('submit-btn').addEventListener('click', function() {
        const queryText = document.getElementById('query-input').value;
        fetch('/query', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ query_text: queryText })
        })
        .then(response => response.json())
        .then(data => {
            // Use innerHTML to render HTML content
            document.getElementById('output').innerHTML = data.message;
        })
        .catch(error => {
            document.getElementById('output').innerText = 'Error occurred while processing the query.';
        });
    });
});
