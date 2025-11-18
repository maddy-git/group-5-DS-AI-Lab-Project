# API Documentation

The RAG pipeline can be accessed programmatically via a Flask endpoint instead of the Streamlit UI.

---

## Endpoint
- Path: `/query`
- Method: `POST`
- Content-Type: `application/json`

---

## Example cURL Request
```bash
curl -X POST http://localhost:5000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "your test question here"}'
```

---

## Input Format
```json
{
  "query": "I need a blue jean"
}
```

---

## Output Format
```json
{
  "response": "llm_response (may contain policy-related answers, product suggestions, etc)",
  "images": [
    "path/to/suggested/product/image1.jpg",
    "path/to/suggested/product/image2.jpg"
  ]
}
```

