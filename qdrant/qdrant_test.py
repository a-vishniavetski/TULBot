from importing_data_to_qdrant import *

# Example search:
# # Prepare the query vector (example: a random vector of dimension 768)
# query_vector = np.random.random(768).tolist()
#
# # Perform the search using the correct method: search()
# search_result = client.query_points(
#     collection_name=collection_name,
#     query=query_vector,
#     limit=5,  # Retrieve 5 closest points
# )
#
# print(search_result)

# # EXAMPLE SEARCHES ON QDRANT DB  # #


# Step 1: Encode the query
query_text = "bazy danych"
inputs = tokenizerA(query_text, return_tensors='pt', padding=True, truncation=True, max_length=512)

with torch.no_grad():
    outputs = modelA(**inputs)

# Get the [CLS] token vector
query_vector = outputs.last_hidden_state[:, 0, :].squeeze().tolist()

# Step 2: Search Qdrant
search_result = client.search(
    collection_name=collection_name,
    query_vector=query_vector,
    limit=5,
    search_params=SearchParams(hnsw_ef=128, exact=False)
)

# Step 3: Print results
for i, result in enumerate(search_result, 1):
    payload = result.payload
    print(f"\nResult {i}:")
    print(f"Subject Name: {payload.get('subject_name')}")
    print(f"Subject ID: {payload.get('subject_id')}")
    print(f"Major: {payload.get('major')}")
    print(f"Content Preview: {payload.get('subject_content', '')}...")

# # Searching for one definite subject
#
# exact_query = "Etyka biznesu"
#
#
# # Define exact filter on 'subject_name'
# search_filter = Filter(
#     must=[
#         FieldCondition(
#             key="subject_name",
#             match=MatchValue(value=exact_query)
#         )
#     ]
# )
# # Run the search with a dummy vector (can be zeroed if exact match is only based on filter)
# dummy_vector = [0.0] * model.config.hidden_size  # Or your known embedding size (e.g., 768)
#
# search_result = client.search(
#     collection_name=collection_name,
#     query_vector=dummy_vector,
#     query_filter=search_filter,
#     limit=1,
#     search_params=SearchParams(hnsw_ef=128, exact=True)
# )
#
# # Check and print results
# if search_result:
#     subject = search_result[0].payload
#     print()
#     print("✅ Subject: ", exact_query, "exists within the database")
#     print(f"Subject ID: {subject.get('subject_id')}")
#     print(f"Major: {subject.get('major')}")
# else:
#     print("❌ Did not find ", exact_query)