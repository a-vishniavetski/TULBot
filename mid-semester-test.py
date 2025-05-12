import json
import torch
from transformers import AutoTokenizer, AutoModel
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance

# Load JSON data
json_file_path = 'base_major.json'
print("Loading data...")
with open(json_file_path, 'r') as file:
    data = json.load(file)
print(f"Loaded {len(data['semesters'])} semesters.")

# Initialize tokenizer and model
print("Initializing embedding model...")
tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
model = AutoModel.from_pretrained("bert-base-multilingual-cased")

# Qdrant client setup
client = QdrantClient(
    url="https://bc99e4de-626a-4844-bd3c-ddadabe39525.europe-west3-0.gcp.cloud.qdrant.io:6333",
    api_key="xyz",
)
# Create or recreate the collection
collection_name = "university_courses"
print(f"Creating collection '{collection_name}'...")
client.recreate_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=768, distance=Distance.COSINE),
)

# Embedding function
def embed_text(text: str) -> list:
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().tolist()
    return embeddings


# Load base_major.json
with open("base_major.json", "r", encoding="utf-8") as f:
    base_major_data = json.load(f)

# Collect and insert data from both sources
print("Processing and inserting embeddings...")
points = []
point_id = 1

def process_and_append(subjects, point_id_start):
    local_points = []
    local_id = point_id_start
    for subject in subjects:
        subject_name = subject.get("subject_name", "")
        overview = subject.get("subject_overview", {})

        # If subject_overview is a string, treat it as such
        if isinstance(overview, str):
            subject_effects = overview
        else:
            subject_effects = overview.get("subject_effects", "")

        full_text = f"{subject_name} {subject_effects}"

        embedding = embed_text(full_text)
        payload = subject  # Store full subject info (can be customized)

        local_points.append(
            PointStruct(id=local_id, vector=embedding, payload=payload)
        )
        local_id += 1
    return local_points, local_id


# Process subjects from both datasets
for semester in data["semesters"]:
    new_points, point_id = process_and_append(semester["subjects"], point_id)
    points.extend(new_points)

for semester in base_major_data["semesters"]:
    new_points, point_id = process_and_append(semester["subjects"], point_id)
    points.extend(new_points)

# Insert all into Qdrant
print(f"Inserting {len(points)} points into Qdrant...")
client.upsert(
    collection_name=collection_name,
    wait=True,
    points=points
)
print("Insertion completed.")

# Run a sample query
print("Running a sample search query...")
sample_query = embed_text("Machine learning introduction and its applications")
search_result = client.search(
    collection_name=collection_name,
    query_vector=sample_query,
    limit=3,
    with_payload=True
)

# Show results
print("\nTop matching subjects:")
for result in search_result:
    print(f"Score: {result.score:.4f}, Subject: {result.payload.get('subject_name')}")
