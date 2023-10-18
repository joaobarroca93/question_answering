import streamlit as st

from pathlib import Path

from src.clients import ChromaDatabaseClient
from src.encoders import SentenceTransformersEncoder
from src.retriever import VectorSearchRetriever
from src.readers import Reader
from src.pipelines import ExtractiveQAPipeline


MAIN_PATH = Path.cwd().resolve().absolute()
MODELS_PATH = MAIN_PATH / "models"
DB_PATH = MAIN_PATH / "data/chroma_db"
COLLECTION_NAME = "documents-all-mpnet-base-deus"
ENCODER_MODEL_FILEPATH = MODELS_PATH / "encoders/sentence-transformers/all-mpnet-base-v2-deus"
READER_MODEL_FILEPATH = "deepset/roberta-base-squad2"


@st.cache_resource
def initialize_encoder():
    return SentenceTransformersEncoder(model_filepath=ENCODER_MODEL_FILEPATH)


@st.cache_resource
def initialize_reader():
    return Reader(model_filepath=READER_MODEL_FILEPATH)


@st.cache_resource
def initialize_db_client():
    return ChromaDatabaseClient(collection_name=COLLECTION_NAME, persist=True, persist_path=str(DB_PATH))


encoder = initialize_encoder()
reader = initialize_reader()
client = initialize_db_client()
retriever = VectorSearchRetriever(client=client, encoder=encoder)
extractive_qa = ExtractiveQAPipeline(retriever=retriever, reader=reader)

st.title("Extractive QA")

top_k = st.slider(
    "top_k",
    1,
    10,
    3,
)

st.subheader("Examples of Questions")
st.write("Until 1999 what was the Williams Tower known as?")
st.write("When was the Premier League created?")
st.write("Pope Sixtus V limited the number of cardinals to?")

question = st.text_input("Question: ", value="")

if question:
    answers = extractive_qa.run(question, top_k=top_k, include_contexts=True)

    # Weight scores by relevance
    weighted_answers = [
        {"answer": answer.content, "final_score": answer.score * answer.context.relevance} for answer in answers
    ]

    st.write("Question: ", question)
    st.write("Answers: ", sorted(weighted_answers, key=lambda x: -x["final_score"]))

    if st.checkbox("Show contexts:"):
        st.subheader("Contexts")
        st.write([answer.as_dict() for answer in answers])
