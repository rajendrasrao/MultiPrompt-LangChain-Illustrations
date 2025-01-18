from langchain_google_genai import GoogleGenerativeAI
from operator import itemgetter
from typing import Literal

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict
from langchain_google_genai import GoogleGenerativeAI
import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import Literal
from langchain.chains.router.multi_prompt_prompt import MULTI_PROMPT_ROUTER_TEMPLATE
from langchain.chains.router import MultiPromptChain
from langchain.chains import LLMChain
from langchain.chains import ConversationChain
from langchain_core.prompts.chat import ChatPromptTemplate,PromptTemplate
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
import pandas as pd
import streamlit as st



llm_model=GoogleGenerativeAI(model="gemini-1.5-flash",temperature=0.0)
df=pd.read_json("saved_data.json")
destinations = [f"{getattr(p, 'name')}: {getattr(p, 'description')}" for p in df.itertuples(index=False)] 
# for all destination I have to add add header for this page to show what topic this chatbot serve.
#I will take help from LLM
print("topic summarys:",destinations)
header_template=PromptTemplate.from_template("""I am creating chatbot with various topics. 
                               You have to create small descriptive title of my chatbot page showing its capablity please provide only one title and no verbose.  topic name and descriptions are:{desc} """)    
headerchain=header_template|llm_model
response_header=headerchain.invoke({"desc":destinations}) 

def prepare_get_chain():
    destination_chains = {}
    for index, row in df.iterrows():
        name = row["name"]
        prompt_template = row["prompt_template"]
        prompt = PromptTemplate(template=prompt_template, input_variables=["input"])
        chain = LLMChain(llm=llm_model, prompt=prompt)
        destination_chains[name] = chain
    default_chain = ConversationChain(llm=llm_model, output_key="text")
    destinations = [f"{getattr(p, 'name')}: {getattr(p, 'description')}" for p in df.itertuples(index=False)] 
    destinations_str = "\n".join(destinations)
    print("destination prompts", destinations_str)
    router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(destinations=destinations_str)
    router_prompt = PromptTemplate(
        template=router_template,
        input_variables=["input"],
        output_parser=RouterOutputParser(),
    )
    router_chain = LLMRouterChain.from_llm(llm_model, router_prompt)

    chain = MultiPromptChain(
        router_chain=router_chain,
        destination_chains=destination_chains,
        default_chain=default_chain,
        verbose=True,
    )
    return chain
    print("******chain created")

st.set_page_config(
    page_title="chatbot for multi expert. (please set prompt info for all desired experts in init_add_prompt.py )",
    page_icon="",
    layout="wide",
)


st.header(response_header)
text_input = st.text_area("Enter your text here:", height=200)
submit_button = st.button("Submit")
chain=prepare_get_chain()

if submit_button:

    # Process the text here (e.g., clean, transform, analyze)
    response=chain.run(text_input)
    # Display the processed text
    st.subheader("Answer:")
    st.text_area(response, height=200)
    
