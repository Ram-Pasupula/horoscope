import logging
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores.redis import Redis
from langchain.llms import HuggingFacePipeline

from transformers import (
    GenerationConfig,
    pipeline,
)
from load_models import (
    load_mps_model,
    load_o_model,
)
from constants import (
    EMBEDDING_MODEL_NAME,
    MODEL_ID,
    MODEL_BASENAME,
    MAX_NEW_TOKENS,
)

def load_model(device_type, model_id, model_basename=None, LOGGING=logging):
    logging.info(f"Loading Model: {model_id}, on: {device_type}")
    logging.info("This action can take a few minutes!")
    #logging.info("model_basename: " + model_basename)
    if model_basename is not None:
        if ".gguf" in model_basename.lower():
            llm = load_mps_model(model_id, model_basename, device_type, LOGGING)
            return llm
    else:
        model, tokenizer = load_o_model(model_id, model_basename, device_type, LOGGING)

    # Load configuration from the model to avoid warnings
    generation_config = GenerationConfig.from_pretrained(model_id)
   
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=MAX_NEW_TOKENS,
        temperature=0.5,
        # top_p=0.95,
        repetition_penalty=1,
        generation_config=generation_config,
    )

    local_llm = HuggingFacePipeline(pipeline=pipe)
    logging.info("Local LLM Loaded")

    return local_llm

llm = load_model('mps', model_id=MODEL_ID, model_basename=MODEL_BASENAME, LOGGING=logging)

embeddings = HuggingFaceInstructEmbeddings(model_name=EMBEDDING_MODEL_NAME ,
                                           model_kwargs={"device": 'mps'},
   # cache_folder = EMBED_CACHE_FOLDER
    )
 
vectorstore = Redis(
        embedding_function=embeddings.embed_query,
        redis_url="redis://localhost:6379",
        index_name="ccda1",
)

retriever = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(),
    # we'll want to see source documents for comparison
    return_source_documents=True
    
)

def print_result(response_obj):
    print("SOURCES: \n")
    cnt = 1
    for source_doc in response_obj["source_documents"]:
        print(f"Chunk #{cnt}")
        cnt += 1
        print("Source Metadata: ", source_doc.metadata)
        print("Source Text:")
        print(source_doc.page_content)
        print("\n")
    print("RESULT: \n")
    #print(response_obj["result"] + "\n\n")

query = "Patient Isabella demographic details?"
response = retriever({"query":query})
print_result(response)