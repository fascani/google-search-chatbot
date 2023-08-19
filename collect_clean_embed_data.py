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
from google.oauth2 import service_account
import gspread

# 1. Collect the results from a google search
#############################################
def collect_links(google_search, serp_api_token):
  '''
  Collect links from a google search

  Parameters
  ----------
  google_search : str
      Google search string
  serp_api_token : str
      API token for SerpApi. See https://serpapi.com/

  Returns
  -------
      List of URLs (str).
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
  
def save_into_google_sheet(texts):
  '''
  Save all pieces of visible texts into the Google sheet.

  Parameters
  ----------
  texts : List of str
    List of pieces of texts extracted from the urls.
  '''
  sheet = access_sheet('info')
  for tt, text in enumerate(texts):
      sheet.update_cell(tt+2, 1, text)
