import sys
import logging
import click
import torch
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter
from langchain.vectorstores.redis import Redis
from langchain.document_loaders import DirectoryLoader, TextLoader
# PDF Loaders
from langchain.document_loaders import PyPDFLoader
import numpy as np
import os

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
    loader = DirectoryLoader(f'{SOURCE_DIRECTORY}',
                             glob="./*.pdf", loader_cls=PyPDFLoader, show_progress=True)
    data = loader.load()
    print(f'You have {len(data)} document(s) in your data')
    print(
        f'There are {len(data[len(data)-1].page_content)} characters in your document')
    # Create embeddings
    embeddings = HuggingFaceInstructEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": device_type},
        cache_folder=EMBED_CACHE_FOLDER
    )

    python_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000, chunk_overlap=1
    )
    data_chunks = np.array_split(data, 50)
    for chunk in data_chunks:
        #print(chunk)
        #print(type(chunk))
        texts = python_splitter.split_documents(chunk)
        print(f'Now you have {len(texts)} documents')
        print(texts[0])
        logging.info("load documents into Redis db")

        rds = Redis.from_documents(
            texts,
            embeddings,
            redis_url="redis://localhost:6379",
            index_name="vedic",
        )
        #break
    logging.info("search documents from Redis db")

    results = rds.similarity_search("What does Rahu do in the second house?")
    logging.info(results[0])
    #print(results[0].page_content[:450])


if __name__ == "__main__":
    main()
