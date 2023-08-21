#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Aug 20, 2023

@author: francoisascani
"""

import os
import streamlit as st
import pandas as pd
#from oauth2client.service_account import ServiceAccountCredentials
from google.oauth2 import service_account
import gspread
import openai
import numpy as np
from transformers import GPT2TokenizerFast
from sentence_transformers import SentenceTransformer
import datetime
from streamlit_chat import message

# Streamlit app
###############

username = 'user'

# from https://docs.streamlit.io/knowledge-base/deploy/authentication-without-sso#option-2-individual-password-for-each-user
def check_password():
    """Returns `True` if the user had a correct password."""

    st.set_page_config(page_title="Hello! Welcome to Francois Ascani's chatbot")
    st.title("Hello! Welcome to Francois Ascani's chatbot")
    
    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if (
            st.session_state["username"] in st.secrets["passwords"]
            and st.session_state["password"]
            == st.secrets["passwords"][st.session_state["username"]]
        ):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store username + password
            #del st.session_state["username"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show inputs for username + password.
        st.text_input("Username", on_change=password_entered, key="username")
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        st.text_input("Username", on_change=password_entered, key="username")
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 User not known or password incorrect")
        return False
    else:
        # Password correct.
        return True

if check_password():
    
    if 'kept_username' not in st.session_state:
        st.session_state['kept_username'] = st.session_state['username']

    # (adapted from https://medium.com/@avra42/build-your-own-chatbot-with-openai-gpt-3-and-streamlit-6f1330876846)
    #st.set_page_config(page_title="Ask Me Anything (AMA), Francois Ascani's chatbot")
    st.title("Curology Product Chatbot")
    st.subheader("A chatbot on internet reviews about Curology's products")
    st.markdown("Aloha! I am Sarah, a chatbot with access to a database of reviews on your skincare products collected from the internet.")
    st.markdown("The workflow has two steps. In the first step, I have collected and cleaned the text from review websites, calculated the 'embeddings' (aka the vectorization) of each piece of text and clustered them into a dozens of groups (aka indexing).")
    st.markdown("In the second step, I calculate the embeddings of your query and find the cluster that is most relevant to your question. I then add the content of the cluster to the prompt (aka 'in-context learning') before I answer your question by calling for OpenAI's chatGPT via an API.")
                
    # Read database on Google sheet
    ###############################
    @st.cache_resource
    def access_sheet(sheet_name):
        '''
        Access the Google's spreadsheet. 

        See https://docs.streamlit.io/knowledge-base/tutorials/databases/private-gsheet
        '''
        # From local computer
        scope = ['https://spreadsheets.google.com/feeds',
                 'https://www.googleapis.com/auth/drive']
        credentials = service_account.Credentials.from_service_account_info(
                    st.secrets["gcp_service_account"], scopes = scope)
        gc = gspread.authorize(credentials)
        sheet = gc.open('review-on-curology-chatbot-db').worksheet(sheet_name)
        return sheet

    def parse_numbers(s):
        if s != '':
            return [float(x) for x in s.strip('[]').split(',')]
        else:
            return ''

    @st.cache_data
    def get_data():
        '''
        Read the database created from the Google search about reviews

        Returns
        -------
        df : Pandas dataframe
            Contains columns 'text', 'num_tokens', 'embeddings', 'cluster' & 'outlier'

        '''
        sheet = access_sheet('info')
        data = sheet.get_all_values()
        df = pd.DataFrame(data[1:], columns=['text', 'num_tokens', 'embeddings', 'cluster', 'outlier'])
        for col in ['embeddings']:
            df[col] = df[col].apply(lambda x: parse_numbers(x))
        for col in ['num_tokens', 'cluster']:
            df[col] = df[col].apply(lambda x: int(x) if x != '' else '')
        return df

    
    # Function to calculate embeddings
    ##################################

    # Set the OpenAI API key
    openai.api_key = st.secrets["openai_api_key"]

    def get_embeddings(text):
        '''
        Calculate embeddings.

        Parameters
        ----------
        text : str
            Text to calculate the embeddings for.

        Returns
        -------
            List of the embeddings
        '''

        model = 'text-embedding-ada-002'
        result = openai.Embedding.create(
          model=model,
          input=text
        )
        embedding = result["data"][0]["embedding"]
        
        return embedding

    # Order the entries by how relevant they are to a string query
    ##############################################################
    def vector_similarity(x, y):
        '''
        Calculate the dot product between two vectors.

        Parameters
        ----------
        x : Numpy array
        y : Numpy array

        Returns
        -------
        Float
            Dot product

        '''
        return np.dot(np.array(x), np.array(y))

    def order_entries_by_similarity(query, df):
        '''
        Calculate the similarity measure for each entry compared to
        a given query.

        Parameters
        ----------
        query : str
            Query.
        df : Pandas dataframe
            Entries with embeddings
  
        Returns
        -------
        df : Pandas dataframe
            Entries with a new column 'similarity'.

        '''
        query_embedding = get_embeddings(query)
        df['similarity'] = df['embeddings'].apply(lambda x: vector_similarity(x, query_embedding))
        df.sort_values(by='similarity', inplace=True, ascending=False)
        df.reset_index(drop=True, inplace=True)

        return df

    # Construct the prompt
    ######################

    # Set the tokenizer
    @st.cache_resource
    def load_tokenizer():
        return GPT2TokenizerFast.from_pretrained("gpt2")

    def get_max_num_tokens():
        '''
        Max number of tokens a pre-trained NLP model can take.
        '''
        return 2046

    def construct_prompt(query, df):
        '''
        Construct the prompt to answer the query. The prompt is composed of the
        query (from the user) and a context containing  the entries that are the
        most relevant (similar) to the query.

        Parameters
        ----------
        query : str
            Query.
        df : Pandas dataframe
            Entries with embeddings.
        
        Returns
        -------
        prompt : str
            Prompt.
        '''

        MAX_SECTION_LEN = get_max_num_tokens()
        SEPARATOR = "\n* "
        tokenizer = load_tokenizer()
        separator_len = len(tokenizer.tokenize(SEPARATOR))

        chosen_sections = []
        chosen_sections_len = 0

        # Order df by their similarity with the query
        df = order_entries_by_similarity(query, df)

        for section_index in range(len(df)):
            # Add contexts until we run out of space.        
            document_section = df.loc[section_index]

            chosen_sections_len += document_section.num_tokens + separator_len
            if chosen_sections_len > MAX_SECTION_LEN:
                break

            chosen_sections.append(SEPARATOR + document_section.text.replace("\n", " "))

        header = """
        You are conversing with a product manager or an executive from a start-up about skin care products. The context
        given are from a Google search on reviews about this start-up's products.
        Context:\n
        """
        prompt = header + "".join(chosen_sections) + "\n\n Q: " + query + "\n A:"

        return prompt

    def record_question_answer(user, query, answer):
        '''
        Record the query, prompt and answer in the database
        '''
        sheet = access_sheet('Q&A')
        # Read how many records we have
        data = sheet.get_all_values()
        df = pd.DataFrame(data[1:], columns=['user', 'date', 'query', 'answer'])
        num_records = len(df)
        today_str = datetime.datetime.strftime(datetime.datetime.today(), '%Y-%m-%d')
        sheet.update_cell(num_records+2, 1, user)
        sheet.update_cell(num_records+2, 2, today_str)
        sheet.update_cell(num_records+2, 3, query)
        sheet.update_cell(num_records+2, 4, answer)

    def google_search_chatbot(query, df):
        '''
        Use a pre-trained NLP method to answer a question given a database
        of information.

        The function also records the query, the prompt, and the answer in
        the database.

        Parameters
        ----------
        query : str
            Query
        df : Pandas dataframe
            Entries with embeddings.
        
        Returns
        -------
        answer : str
            Answer from the model.
        prompt : str
            Actual prompt built.

        '''

        # Construct the prompt
        prompt = construct_prompt(query, df)

        # Ask the question with the context with GPT3 text-davinci-003
        COMPLETIONS_MODEL = "text-davinci-003"

        response = openai.Completion.create(
            prompt=prompt,
            temperature=0.9,
            max_tokens=300,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            model=COMPLETIONS_MODEL
        )

        answer = response["choices"][0]["text"].strip(" \n")

        return answer, prompt
    
    # Prepare engine
    method = 'openai'
    df = get_data()    
    
    # Storing the chat
    if 'generated' not in st.session_state:
        st.session_state['generated'] = []

    if 'past' not in st.session_state:
        st.session_state['past'] = []

    # Get user's input
    hello_message = "Hello, how are you? Ask questions about your skin care products or skin care in general."
    def get_text(hello_message):
        input_text = st.text_input("You: ", hello_message, key="input")
        return input_text

    user_input = get_text(hello_message)

    # Get the answer
    if user_input:
        answer, prompt = ama_chatbot(user_input, df, method)
        # Store the output 
        st.session_state.past.append(user_input)
        st.session_state.generated.append(answer)
        # Record the interaction if not the hello message
        if user_input != hello_message:
            record_question_answer(st.session_state['kept_username'], user_input, answer)

    # Display the chat    
    if st.session_state['generated']:

        for i in range(len(st.session_state['generated'])-1, -1, -1):
            message(st.session_state["generated"][i], key=str(i), avatar_style="bottts")
            message(st.session_state['past'][i], is_user=True, key=str(i) + '_user', avatar_style='human')
