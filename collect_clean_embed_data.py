#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
In this code,
1. We collect the results from a google search
2. We parse the html from each link and return the visible text
3. We use chatGPT to decide whether to keep each piece of text in reference to the topic
4. We calculate the embeddings of each piece of text that was kept
5. We create a csv file of the text and its embedding
"""

import os
from serpapi import GoogleSearch
from bs4 import BeautifulSoup
import requests
import openai

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


def answer_Y_N(text):
    # Ask the question with the context with GPT3 text-davinci-003
    COMPLETIONS_MODEL = "text-davinci-003"

    prompt = """
    Does this piece of text is about a review: "<text>".
    Answer only Y or N.
    """.replace('<text>', text)
    
    response = openai.Completion.create(
        prompt=prompt,
        temperature=0,
        max_tokens=1580,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        model=COMPLETIONS_MODEL
    )

    answer = response["choices"][0]["text"].strip(" \n")

    return answer
