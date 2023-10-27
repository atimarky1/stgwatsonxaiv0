import streamlit as st
st.set_page_config(page_title="app", page_icon="😊", layout="wide")
import pandas as pd
import plotly.graph_objects as go
from PIL import Image
from streamlit_elements import elements, mui, html
import locale
from streamlit_extras.add_vertical_space import add_vertical_space
import matplotlib.pyplot as plt
from streamlit_card import card
from streamlit_extras.metric_cards import style_metric_cards 
import plotly.express as px
import os
import streamlit_authenticator as stauth
from pathlib import Path
from dotenv import load_dotenv

import pickle
import requests
import json
from ibm_cloud_sdk_core import IAMTokenManager
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator, BearerTokenAuthenticator
import os
from utils import (
    embed_docs_v2,
    get_sources_v2,
    parse_docx,
    parse_pdf_v2,
    parse_txt,
    search_docs_v2,
    text_to_docs,
    wrap_text_in_html,
    parse_pdf_v3
)
from prompts import get_model_output, get_model_output_rag, generate_prompt_rag, get_model_output_rag_qa


load_dotenv()

    # Update the global variables that will be used for authentication in another function
api_key = st.secrets["api_key"]
project_id = st.secrets["project_id"]

#set_ibm_api_key(api_key, project_id)

try:
    access_token = IAMTokenManager(
        apikey = api_key,
        url = "https://iam.cloud.ibm.com/identity/token"
    ).get_token()
except:
    raise ValueError("Authentication failed!")


def clear_submit():
    st.session_state["submit"] = False

##############################################################################################

pred_output_col_1, pred_output_col_2 = st.columns(2)

with pred_output_col_1:
    st.title("Recent customer interaction:")
with pred_output_col_2:
    st.title("Summary:")

pred_output_img_col, pred_output_chart_col = st.columns(2)

with pred_output_img_col:
    text_input = st.text_area(label = "", value="""[Customer Service Representative]: Thank you for calling [Company Name], my name is John. How can I assist you today?

[Customer]: Hi John, I'm calling because I'm really frustrated with the service, and I'm considering canceling it.

[Customer Service Representative]: I'm sorry to hear that you're feeling this way. Can you please share more about the issue you're facing so that I can try to help resolve it for you?

[Customer]: The problem is that the price of my service is just too high, and I can't afford it anymore. It's been increasing steadily, and it's become a financial burden for me.

[Customer Service Representative]: I understand your concern about the pricing. Let me see if I can assist you in finding a more cost-effective solution. Can you please provide me with your account number or the phone number associated with your account so I can look into your billing history?

[Customer]: Sure, it's [Account Number/Phone Number].

[Customer Service Representative]: Thank you for providing that information. Let me review your account.

[Customer]: Take your time.

[Customer Service Representative]: I've reviewed your account, and I can see that your monthly charges have indeed increased over time. I can understand how that can be frustrating. Can you tell me which specific services or features are causing the price increase?

[Customer]: Well, I initially had a basic package, but I added a few premium channels and upgraded my internet speed. While I understand that these additions come with an extra cost, I just didn't expect the overall price to go up so much.

[Customer Service Representative]: I appreciate your explanation, and I can see how those additions can affect the cost. Let me see if there are any promotions or packages available that could better suit your needs.

[Customer]: That would be great. If we can find a more affordable option, I'd be willing to stay with the company.

[Customer Service Representative]: I appreciate your willingness to explore alternatives. Let me check for available packages. Please hold for a moment while I do that.

[Customer]: Okay, I'll wait.

[Customer Service Representative]: Thank you for waiting. I've searched for alternative packages, but it appears that there are no available promotions or plans that would significantly reduce your monthly bill while maintaining the services you want.

[Customer]: I see. That's disappointing.

[Customer Service Representative]: I understand how you feel, and I'm sorry that I couldn't provide a more cost-effective solution in this case. I know that the price increase can be frustrating. If there's anything else you'd like to discuss or if you're still considering canceling, please let me know how you'd like to proceed.

[Customer]: Well, I appreciate your help, but it seems like I have no other option at this point. I'd like to go ahead and cancel the service.

[Customer Service Representative]: I understand your decision, and I'm sorry that we couldn't find a more suitable solution. I'll process your cancellation request for you. Please keep in mind that any equipment returns or final bills will be part of the cancellation process. Thank you for being a customer, and I wish you the best in your future service needs.

[Customer]: Thank you for your assistance, John. I hope I can find a more affordable option elsewhere.

[Customer Service Representative]: You're welcome, and I appreciate your understanding. If you ever decide to return in the future or have any questions, don't hesitate to reach out. Have a good day.
""", 
                height = 300, on_change=clear_submit)
    summarize_button = st.button("Summarize!")


with pred_output_chart_col:
    if summarize_button:
        prompt_input = "summarize this customer review in 50 words: " + text_input + "\n" + "Summary: "
        output = get_model_output(access_token, prompt_input, 50, 20, project_id)
        prompt_input_2 = "Classify this review as positive or negative. " + text_input + "\n" + "Summary: "
        output_2 = get_model_output(access_token, prompt_input_2, 1, 1, project_id)
        st.write(output)
    st.title("Sentiment")
    if summarize_button:
        if output_2 == "negative":
            st.write(output_2 + "🙁")
        if output_2 == "positive":
            st.write(output_2 + "😃" )


st.markdown("___")

ask_questions_row, = st.columns(1)
with ask_questions_row:
    st.title("Knowledge Assistant")

#####


index = None
doc = []
file_path = "./data/Customer Retention Offers.docx"

doc.append(parse_docx(file_path))

text = text_to_docs(doc)
    #print(text)

with st.spinner("Indexing document... This may take a while⏳"):
    index = embed_docs_v2(text)

try:
    query = st.text_area("Ask a question about the document", on_change=clear_submit, value="Generate an email to a customer with a few offers he is qualified for. " + output)
except:
    query = st.text_area("Ask a question about the document", on_change=clear_submit, value="Generate an email or ask a question here.")

with st.expander("Document"):
        # Hack to get around st.markdown rendering LaTeX
    st.markdown(f"<p>{wrap_text_in_html(doc)}</p>", unsafe_allow_html=True)

button = st.button("Submit")

if button or summarize_button:
    #if not index:
    #    st.error("Please upload a document!")
    if not query:
        st.error("Please enter a question!")
    else:
        st.session_state["submit"] = True
        # Output Columns
        answer_col, sources_col = st.columns(2)
        sources = search_docs_v2(index, query, text)
        try:
        
            prompt_input = generate_prompt_rag(query, sources)
            print("***prompts***")
            print(prompt_input)
        

            answer_col, sources_col = st.columns(2)
            # Get the sources for the answer
            with answer_col:
                st.markdown("#### Answer")
                if "email" in query:
                    answer = get_model_output_rag(access_token, prompt_input, 500, 0, 7, project_id)
                    answer = answer.replace(":", ": ")
                    st.markdown(answer)
                else:
                    answer = get_model_output_rag_qa(access_token, prompt_input, 100, 50, 1482679647, project_id)
                    print(answer)
                    st.markdown(answer)
                    #print(len(answer.split(" ")))

            with sources_col:
                st.markdown("#### Sources")
                for source in sources:
                    st.markdown(source)
                    st.markdown(get_sources_v2(source, text))
                    st.markdown("---")
        except:
            print("generate_prompt_rag error!")
            