from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from typing import List
import json
from cassandra.auth import PlainTextAuthProvider
from cassandra.cluster import Cluster
from llama_index import ServiceContext, set_global_service_context
from llama_index import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.embeddings import GradientEmbedding
from llama_index.llms import GradientBaseModelLLM
from copy import deepcopy
from tempfile import NamedTemporaryFile

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    with open("ai2.html", "r") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

def create_datastax_connection():
    cloud_config = {'secure_connect_bundle': 'secure-connect-temp-db.zip'}

    with open("temp_db-token.json") as f:
        secrets = json.load(f)

    CLIENT_ID = secrets["clientId"]
    CLIENT_SECRET = secrets["secret"]

    auth_provider = PlainTextAuthProvider(CLIENT_ID, CLIENT_SECRET)
    cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)
    astra_session = cluster.connect()
    return astra_session

@app.post("/upload_pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    # Process the uploaded PDF file here
    # For example, save it to a specific location or perform other operations
    return {"detail": "File uploaded successfully"}

@app.post("/send_message/")
async def send_message(message: str):
    # Connect to Cassandra
    session = create_datastax_connection()

    # Set up AI model and service context
    GRADIENT_ACCESS_TOKEN = "GRADIENT_ACCESS_TOKEN"
    GRADIENT_WORKSPACE_ID = "GRADIENT_WORKSPACE_ID"

    llm = GradientBaseModelLLM(base_model_slug="llama2-7b-chat", max_tokens=400)

    embed_model = GradientEmbedding(
        gradient_access_token=GRADIENT_ACCESS_TOKEN,
        gradient_workspace_id=GRADIENT_WORKSPACE_ID,
        gradient_model_slug="bge-large"
    )

    service_context = ServiceContext.from_defaults(
        llm=llm,
        embed_model=embed_model,
        chunk_size=256
    )

    set_global_service_context(service_context)

    # Here you can perform AI-related actions using the 'service_context'
    # and interact with Cassandra using the 'session'

    # Process the message and return the response
    # Replace this with your logic
    return {"response": f"Received message: {message}"}
