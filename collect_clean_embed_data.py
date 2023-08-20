#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
In this code,
1. We collect the results from a google search
2. We parse the html from each link and return the visible text
3. We clean up the text and return only those with a minimum number of tokens
4. Calculate embeddings and organize results into a Pandas df
5. We save the result into a Google sheet
"""

import os
from serpapi import GoogleSearch
from bs4 import BeautifulSoup
import requests
from google.oauth2 import service_account
import gspread
import nltk
import numpy as np
import openai
import pandas as pd
from transformers import GPT2TokenizerFast

# 1. Collect the results from a google search
#############################################
def collect_links(google_search, serp_api_token, start):
  '''
  Collect links from a google search

  Parameters
  ----------
  google_search : str
      Google search string
  serp_api_token : str
      API token for SerpApi. See https://serpapi.com/
  start: int
      Return outputs after the (start)th
  Returns
  -------
      List of URLs (str).
  '''
  results = dict()
  
  params = {
      "engine": "google",
      "q": google_search,
      "api_key": serp_api_token,
      "start": start,
      "num": 10,
  }
    
  search = GoogleSearch(params)
  results.update(search.get_dict())
   
  return results

# 2. We parse the html from each link and return the visible text
#################################################################
def parse_return_texts(results):
  '''
  Return all pieces of visible texts from the list of urls returned by the google search.

  Parameters
  ----------
  results : List of str
    List of urls.

  Returns
  -------
      List of visible texts separated.
  '''
  texts = []
  
  for result in results["organic_results"]:
    print(f"Link: {result['link']}")
    url_link = result['link']
    html = requests.get(url_link).text
    soup = BeautifulSoup(html, 'html.parser')
    new_texts = soup.getText("|").split('|')

  texts.extend(new_texts)

  return texts

# 3. We clean up the text and return only those with a minimum number of tokens
###############################################################################
def clean_texts(texts, min_num_tokens):
  kept_texts = []
  for text in texts:
      # Clean text
      cleaned_text = ' '.join(text.split())
      nltk_tokens = nltk.word_tokenize(cleaned_text)
      if len(nltk_tokens)>=min_num_tokens:
          kept_texts.append(cleaned_text)

  return kept_texts

# 4. Calculate embeddings, number of tokens, and organize results into a Pandas df
##################################################################################
def load_tokenizer():
  return GPT2TokenizerFast.from_pretrained("gpt2")
  
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

def build_df_with_embeddings(texts):
  '''
  Calculate embeddings.

  Parameters
  ----------
  texts : list of str
      List of pieces text
      
  Returns
  -------
      Pandas df with original text, number of tokens and their embeddings
  '''

  embeddings = []
  tokenizer = load_tokenizer()
  num_tokens = []
  
  for text in texts:
    embeddings.append(get_embeddings(text))
    num_tokens.append(len(tokenizer.encode(text)))
    
  df = pd.DataFrame({'text': texts, 'num_tokens': num_tokens, 'embeddings': embeddings})

  return df

# 5. We save the results into a Google sheet
############################################
def access_sheet(service_account_json, google_file_name, sheet_name):
  '''
  Access the Google's spreadsheet. 

  See https://docs.streamlit.io/knowledge-base/tutorials/databases/private-gsheet
  '''
  scope = ['https://spreadsheets.google.com/feeds',
           'https://www.googleapis.com/auth/drive']
  credentials = service_account.Credentials.from_service_account_file(service_account_json, scopes = scope)
  gc = gspread.authorize(credentials)
  sheet = gc.open(google_file_name).worksheet(sheet_name)
  return sheet
  
def save_into_google_sheet(df, sheet, start):
  '''
  Save all pieces of visible texts and their embeddings into the Google sheet.

  Parameters
  ----------
  df : Pandas df
    Pandas df containing the suite of texts and their embeddings.
  start: int
    Row number we start to fill up the sheet
  '''
  for i in range(len(df)):
      sheet.update_cell(start+i+2, 1, df.loc[i, 'text'])
      sheet.update_cell(start+i+2, 2, str(df.loc[i, 'num_tokens']))
      sheet.update_cell(start+i+2, 3, str(df.loc[i, 'embeddings']))
