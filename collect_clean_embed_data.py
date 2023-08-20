#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
In this code,
1. We collect the results from a google search
2. We parse the html from each link and return the visible text
3. We clean up the text and return only those with a minimum number of tokens
4. We save the result into a Google sheet
"""

import os
from serpapi import GoogleSearch
from bs4 import BeautifulSoup
import requests
from google.oauth2 import service_account
import gspread
import nltk
import numpy as np

# 1. Collect the results from a google search
#############################################
def collect_links(google_search, serp_api_token, num):
  '''
  Collect links from a google search

  Parameters
  ----------
  google_search : str
      Google search string
  serp_api_token : str
      API token for SerpApi. See https://serpapi.com/
  num: int
      Number of links to return
  Returns
  -------
      List of URLs (str).
  '''
  results = dict()
  start = 0
  KeepGoing = True

  while KeepGoing:
    params = {
        "engine": "google",
        "q": google_search,
        "api_key": serp_api_token,
        "start": start,
        "num": np.min(10, num-start),
    }
    
    search = GoogleSearch(params)
    results.update(search.get_dict())
    if len(results) > num:
      KeepGoing = False
    else:
      start += 10
  
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

# 3. We save the result into a Google sheet
###########################################
def clean_texts(texts, min_num_tokens):
  kept_texts = []
  for text in texts:
      # Clean text
      cleaned_text = ' '.join(text.split())
      nltk_tokens = nltk.word_tokenize(cleaned_text)
      if len(nltk_tokens)>=min_num_tokens:
          kept_texts.append(cleaned_text)

  return kept_texts

# 4. We save the result into a Google sheet
###########################################
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
  
def save_into_google_sheet(texts, sheet):
  '''
  Save all pieces of visible texts into the Google sheet.

  Parameters
  ----------
  texts : List of str
    List of pieces of texts extracted from the urls.
  '''
  for tt, text in enumerate(texts):
      sheet.update_cell(tt+2, 1, text)
