import os
import sys
import time
import logging
import click
import torch
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.llms import HuggingFacePipeline
from pt_utils import get_prompt_template
import streamlit as st
from langchain.vectorstores.redis import Redis
from load_fe_model import load_mps_model, load_o_model, load_fe_model
from constants import (
    EMBEDDING_MODEL_NAME,
    MODEL_ID,
    MODEL_BASENAME,
    MAX_NEW_TOKENS,
    MODELS_PATH,
    EMBED_CACHE_FOLDER
)
from style_util import (change_label_style, TOOL_HIDE, get_footer,
                        get_page_conf, label, title, side_foot)
from transformers import (GenerationConfig, pipeline)
os.system('cls' if os.name == 'nt' else 'clear')
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# Instantiate Embeddings instance
if "EMBEDDINGS" not in st.session_state:
    embeddings = HuggingFaceInstructEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": 'mps'},
        cache_folder=EMBED_CACHE_FOLDER,
        encode_kwargs={'normalize_embeddings': True},
        
    )
    st.session_state.EMBEDDINGS = embeddings
# Instantiate DB instance
if "DB" not in st.session_state:
    db = Redis(
        embedding_function=embeddings.embed_query,
        redis_url="redis://localhost:6379",
        index_name="ccda1",
    )
    st.session_state.DB = db

if "RETRIEVER" not in st.session_state:
    retriever = st.session_state["DB"].as_retriever(search_kwargs={"k": 1})
    st.session_state["RETRIEVER"] = retriever

    # get the prompt template and memory if set by the user llama/mistral.
if "PROMPT_MEM" not in st.session_state:
    prompt, memory = get_prompt_template(
        promptTemplate_type="llama", history=True)
    st.session_state["PROMPT"] = prompt
    st.session_state["MEMORY"] = memory

# load the llm pipeline


def load_llm_model(device_type, model_id, model_basename=None, logging=logging):
    logging.info("loading llm model ..!")
    if model_basename is not None:
        if ".gguf" in model_basename.lower():
            llm = load_mps_model(
                model_id, model_basename, device_type, logging)
            return llm
        else:
            model, tokenizer = load_fe_model(model_id, model_basename, device_type, logging)
    else:
        model, tokenizer = load_o_model(model_id, device_type, logging)

    #generation_config = GenerationConfig.from_pretrained(model_id)

    pipe = pipeline(
        "feature-extraction",
        # "question-answering",
        model=model,
        tokenizer=tokenizer,
        #max_length=MAX_NEW_TOKENS,
        temperature=0.2,
        # top_p=0.95,
        # do_sample=True,
        repetition_penalty=1,
        return_full_text=False,
        #generation_config=generation_config,
    )

    local_llm = HuggingFacePipeline(pipeline=pipe)
    logging.info("LLM Loaded")
    return local_llm


if "LLM" not in st.session_state:
    llm = load_llm_model('mps', model_id=MODEL_ID,
                         model_basename=MODEL_BASENAME, logging=logging)
    st.session_state["LLM"] = llm


def retrieval_qa_pipline(use_history):
    if use_history:
        qa = RetrievalQA.from_chain_type(
            llm=st.session_state["LLM"],
            chain_type="stuff",  # refine, map_reduce, map_rerank
            retriever=st.session_state["RETRIEVER"],
            return_source_documents=False,
            chain_type_kwargs={
                "verbose": False,
                "prompt": st.session_state["PROMPT"],
                "memory": st.session_state["MEMORY"]
            },
        )
    else:
        qa = RetrievalQA.from_chain_type(
            llm=st.session_state["LLM"],
            chain_type="stuff",
            retriever=st.session_state["RETRIEVER"],
            return_source_documents=False,
            verbose=False,
            chain_type_kwargs={
                "verbose": False,
                "prompt": st.session_state["PROMPT"],
                "memory": st.session_state["MEMORY"]
            },
        )

    return qa


def submit():
    st.session_state["widget"] = ""


@click.command()
@click.option(
    "--device_type",
    default="mps" if torch.backends.mps.is_available() else "cpu",
    type=click.Choice(
        [
            "cpu",
            "cuda",
            "mps",
        ],
    ),
    help="Device to run on. (Default is cuda)",
)
@click.option(
    "--show_sources",
    "-s",
    is_flag=True,
    default=False,
    help="Show sources along with answers (Default is False)",
)
@click.option(
    "--use_history",
    "-h",
    default=True,
    is_flag=True,
    help="Use history (Default is False)",
)
@click.option(
    "--model_type",
    default="llama",
    type=click.Choice(
        ["llama", "mistral", "non_llama"],
    ),
    help="model type, llama, mistral or non_llama",
)
def main(device_type, show_sources, use_history, model_type):
    logging.info(f"Running on: {device_type}")
    logging.info(f"Display Source Documents set to: {show_sources}")
    logging.info(f"Use history set to: {use_history}")
    if "my_text" not in st.session_state:
        st.session_state.my_text = ""
    # check if models directory do not exist, create a new one and store models here.
    if not os.path.exists(MODELS_PATH):
        os.mkdir(MODELS_PATH)
    if "QA" not in st.session_state:
        QA = retrieval_qa_pipline(use_history)
        st.session_state["QA"] = QA
    # os.system('cls' if os.name == 'nt' else 'clear')
    prompt = st.text_input(f":black[{label}]",  key="widget")
    start_time = time.time()
    if prompt:
        res = st.session_state["QA"](prompt)
        answer = res["result"]
        st.markdown(answer)
        # change_label_style(answer, '10px')
    end_time = time.time()
    st.button("Reset", type="primary", on_click=submit)
    elapsed_time = end_time - start_time
    elapsed_minutes = elapsed_time / 60
    print(f"Execution time: {elapsed_minutes:.2f} minutes")


get_page_conf()
st.markdown(TOOL_HIDE, unsafe_allow_html=True)
get_footer()
# Sidebar contents
with st.sidebar:
    st.title(title)
    st.header(" About")
    st.markdown(side_foot, unsafe_allow_html=True)
    line = "------------------------------"
    st.markdown(line)
    # st.write("------------🚫🔄🚀⚠️💡💸----------------")
if __name__ == "__main__":
    # os.system('cls' if os.name == 'nt' else 'clear')
    main()
