import sys
import logging
import click
import torch
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter
from langchain.vectorstores.redis import Redis

from constants import (
    EMBEDDING_MODEL_NAME,
    SOURCE_DIRECTORY,
    EMBED_CACHE_FOLDER
)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

@click.command()
@click.option(
    "--device_type",
    default="mps" if torch.cuda.is_available() else "cpu",
    type=click.Choice(
        [
            "cpu",
            "cuda",
            "ipu",
            "xpu",
            "mps",
        ],
    ),
    help="Device to run on. (Default is mps)",
)
def main(device_type):
    # Load documents and split in chunks
    logging.info(f"Loading documents from {SOURCE_DIRECTORY}")
    # Create embeddings
    embeddings = HuggingFaceInstructEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": device_type},
        cache_folder=EMBED_CACHE_FOLDER
    )

    loader = DirectoryLoader(
        SOURCE_DIRECTORY,
        glob='**/data1.json',
        show_progress=True,
        loader_cls=TextLoader
    )
    python_documents = loader.load()

    python_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON, chunk_size=800, chunk_overlap=10
    )
    texts = python_splitter.split_documents(python_documents)
    logging.info("load documents into Redis db")
    rds = Redis.from_documents(
        texts,
        embeddings,
        redis_url="redis://localhost:6379",
        index_name="ccdas",
    )
    logging.info("search documents from Redis db")

    results = rds.similarity_search("Patient Isabella demographic details")
    logging.info(results)


if __name__ == "__main__":
    main()
