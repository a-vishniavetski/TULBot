import json
import os
import uuid
from pathlib import Path

import numpy as np
from qdrant_client.grpc import SearchParams
from qdrant_client.models import Filter, FieldCondition, MatchValue
import torch
from torch_geometric.nn.nlp import SentenceTransformer
from transformers import AutoTokenizer, AutoModel, DistilBertTokenizer, DistilBertModel
from qdrant_client.models import PointStruct, VectorParams, Distance
from qdrant_client import QdrantClient

collection_name = "university_subjects"  # Replace with your collection's name

# Initialize tokenizer and model
print("Initializing embedding model...")
tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
model = AutoModel.from_pretrained("bert-base-multilingual-cased")

client = QdrantClient(
    url="https://bc99e4de-626a-4844-bd3c-ddadabe39525.europe-west3-0.gcp.cloud.qdrant.io:6333",  # Replace with your actual URL
    api_key = os.getenv('qdrant_api_key')

)



from transformers import DistilBertTokenizer, DistilBertModel

# Load DistilBERT tokenizer and model from local disk (no Hugging Face API call)
model_path = "distilbert-base-uncased"  # Download it locally if you want to use it offline

tokenizerA = DistilBertTokenizer.from_pretrained(model_path)
modelA = DistilBertModel.from_pretrained(model_path)

# Example sentence
inputs = tokenizer("This is an example sentence.", return_tensors="pt", padding=True, truncation=True)
outputs = model(**inputs)

# Extract the embeddings
embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().tolist()
print(embeddings)

def embed_text(text: str) -> list:
    """Generate embeddings for the input text using DistilBERT model."""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().tolist()
    return embeddings

# Example usage
# text = "This is an example sentence."
# embeddings = embed_text(text)
# print(embeddings)



COLLECTION_NAME = 'university_subjects'

def create_university_subjects_collection():
    """Creates the university_subjects collection in Qdrant."""
    # Get the dimensionality of the hidden states from the model's config
    vector_size = model.config.hidden_size  # Using config.hidden_size to get the dimension

    # Recreate the Qdrant collection with the correct vector size
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
    )
    print(f"Collection '{COLLECTION_NAME}' created with vector size {vector_size}.")

def insert_subject(file_path: str, client: QdrantClient = client, model=model, tokenizer=tokenizer, COLLECTION_NAME=collection_name):
    """
    Vectorizes subject description and inserts it with metadata into Qdrant.
    The data should be loaded from a JSON file at the specified path.
    If the subject already exists, adds new major if not already present.
    """
    # Load the JSON data
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # Extract global metadata
    major = data.get("major", "")
    mode = data.get("mode", "")
    lang = data.get("lang", "")
    semesters = data.get("semesters", [])

    for semester in semesters:
        semester_number = semester.get("semester", "")
        for subject in semester.get("subjects", []):
            subject_name = subject.get("subject_name", "")
            optional_subjects = subject.get("optional_subjects", {}).get("subjects", [])

            if optional_subjects:
                # Process each optional subject
                for opt_subject in optional_subjects:
                    process_subject(
                        opt_subject,
                        major=f"{major} - {subject_name}",
                        mode=mode,
                        lang=lang,
                        semester_number=semester_number,
                        client=client,
                        model=model,
                        tokenizer=tokenizer,
                        COLLECTION_NAME=COLLECTION_NAME
                    )
            else:
                # Regular subject
                process_subject(
                    subject,
                    major=major,
                    mode=mode,
                    lang=lang,
                    semester_number=semester_number,
                    client=client,
                    model=model,
                    tokenizer=tokenizer,
                    COLLECTION_NAME=COLLECTION_NAME
                )

def process_subject(subject, major, mode, lang, semester_number, client, model, tokenizer, COLLECTION_NAME):
    """
    Helper function to process a single subject and insert/update it in Qdrant.
    """
    subject_name = subject.get("subject_name", "")
    overview = subject.get("subject_overview", {})

    subject_id = overview.get("subject_id", "")
    lecture_language = overview.get("lecture_language", "")
    prerequisites = overview.get("prerequisites", "")
    subject_effects = overview.get("subject_effects", "")
    subject_effects_verification = overview.get("subject_effects_verification", "")
    major_study_effects = overview.get("major_study_effects", "")
    subject_content = overview.get("subject_content", "")
    time_distribution = overview.get("time_distribution", {})

    # Vectorize the subject content
    inputs = tokenizer(subject_content, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    vector = outputs.last_hidden_state[:, 0, :].squeeze().tolist()

    # Create filter to check if the subject already exists
    search_filter = Filter(
        must=[
            FieldCondition(
                key="subject_id",
                match=MatchValue(value=subject_id)
            )
        ]
    )

    # Perform the search
    search_results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=vector,
        query_filter=search_filter,
        limit=1,
        search_params=SearchParams(hnsw_ef=128, exact=False)
    )

    if search_results:
        existing_payload = search_results[0].payload
        existing_major = existing_payload.get("major", "")

        if major not in existing_major:
            updated_major = f"{existing_major}, {major}".strip(", ")
            existing_payload["major"] = updated_major

            updated_point = PointStruct(
                id=search_results[0].id,
                vector=vector,
                payload=existing_payload
            )

            client.upsert(
                collection_name=COLLECTION_NAME,
                points=[updated_point]
            )

            print(f"Updated subject '{subject_name}' (ID: {subject_id}) with new major '{major}'.")
        else:
            print(f"Subject '{subject_name}' (ID: {subject_id}) already has the major '{major}'.")
    else:
        payload = {
            "subject_name": subject_name,
            "subject_id": subject_id,
            "major": major,
            "mode": mode,
            "lang": lang,
            "lecture_language": lecture_language,
            "prerequisites": prerequisites,
            "subject_effects": subject_effects,
            "subject_effects_verification": subject_effects_verification,
            "major_study_effects": major_study_effects,
            "subject_content": subject_content,
            "time_distribution": time_distribution,
            "semester": semester_number
        }

        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload=payload
        )

        client.upsert(
            collection_name=COLLECTION_NAME,
            points=[point]
        )

        print(f"Inserted subject '{subject_name}' (ID: {subject_id}) into collection.")



def process_files_in_folder(folder_path: str):
    """
    Process all JSON files in a given folder and insert the subject data into Qdrant.

    :param folder_path: The folder containing JSON files.
    :param client: The Qdrant client instance for interacting with the collection.
    :param model: The model used for embedding generation.
    :param tokenizer: The tokenizer used to process the subject content.
    :param COLLECTION_NAME: The name of the Qdrant collection.
    """
    folder = Path(folder_path)
    json_files = [file for file in folder.glob("*.json")]  # Get all .json files

    for json_file in json_files:
        print(f"Processing file: {json_file.name}")
        try:
            # Pass the file path and other necessary parameters to insert_subject
            insert_subject(json_file)
        except Exception as e:
            print(f"Failed to process {json_file.name}: {e}")


# Example usage
if __name__ == "__main__":
    # create_university_subjects_collection()
    print(client.get_collections())

    collection_name = "university_subjects"  # Replace with your collection's name

    # # Path to your folder with subject JSON files
    # folder_path = "data/FTIMS_majors"  # Replace this with the correct folder path

    # process_files_in_folder(folder_path)
    # insert_subject("data/FTIMS_majors/Aktuariat_i_analiza_finansowa_2024-stacjonarne_pl.json")

