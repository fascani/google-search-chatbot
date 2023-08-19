#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
In this code,
1. We collect the results from a google search
2. We parse the html from each link and return the visible text
3. We save the result into a Google sheet
"""

import os
from serpapi import GoogleSearch
from bs4 import BeautifulSoup
import requests
import openai
from google.oauth2 import service_account

# Tokens
serp_api_token = 'xxxx'
openai_api_token = 'xxxx'

# 1. Collect the results from a google search
#############################################
def collect_links(google_search):
  '''
  '''
  
  params = {
      "engine": "google",
      "q": google_search,
      "api_key": serp_api_token,
  }
  
  search = GoogleSearch(params)
  results = search.get_dict()
  
  return results

# 2. We parse the html from each link and return the visible text
#################################################################
def parse_return_texts(results):
  '''
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
def access_sheet(google_file_name, sheet_name):
    '''
    Access the Google's spreadsheet. 

    See https://docs.streamlit.io/knowledge-base/tutorials/databases/private-gsheet
    '''
    scope = ['https://spreadsheets.google.com/feeds',
             'https://www.googleapis.com/auth/drive']
    credentials = service_account.Credentials.from_service_account_info(
                st.secrets["gcp_service_account"], scopes = scope)
    gc = gspread.authorize(credentials)
    sheet = gc.open(google_file_name).worksheet(sheet_name)
    return sheet
  
def save_into_google_sheet(texts, google_sheet_name):
  
