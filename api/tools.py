import argparse
import logging
import re
from razdel import tokenize, sentenize
from summa.summarizer import summarize
from collections import Counter
from slovnet import Morph
from navec import Navec
from slovnet import NER
from string import punctuation
import numpy as np
import torch
from difflib import SequenceMatcher

logging.basicConfig(
    level=logging.INFO
)

lib_logger = logging.getLogger("russian-question-generation")
lib_logger.info('Starting')


morph = Morph.load('slovnet_morph_news_v1.tar', batch_size=4)
navec = Navec.load('navec_news_v1_1B_250K_300d_100q.tar')
morph.navec(navec)



class TrueFalseArgs():
    temperature: 0.9
    context_size: 5
    max_questions: 10
    topic: ''
    filename: ''
    
    def __init__(self):
      self.parser = argparse.ArgumentParser()
      self.parser.add_argument('-t', '--temperature', default=0.9, type=float, action='store', help='Temperature setting for model')
      self.parser.add_argument('-c', '--context_size', default=5, type=int, action='store', help='Number of sentences used for the context')  
      self.parser.add_argument('-q', '--max_questions', default=10, type=int,action='store', help='Number of questions to generate')  
      self.parser.add_argument('-f', '--filename', default=None, action='store', help='File name of context')  
      self.parser.add_argument('-w', '--topic', default=None, action='store', help='Topic from wikipedia')  
      self.parser.add_argument('-sr', '--summarize_ratio', default=None, type=float, action='store', help='Summarization ratio (for example 0.2). Alternative to --summarize_word_count. Use 1.0 to disable summarization')  
      self.parser.add_argument('-sw', '--summarize_word_count', default=3000,  type=int, action='store', help='Summarization word count (for example 3000). Alternative to --summarize_ratio')  
      self.parser.parse_args(namespace=self)
      
    
class MultipleChoiceArgs():
    context_size:8
    max_questions:10
    summarize_ratio: None
    temperature_answer:0.5
    temperature_question:0.5
    temperature_wrong_answer:2.0
    summarize_word_count:3000
    generate_count: 20
    generate_size: 1
    topic: ''
    filename: ''
    
    def __init__(self):
      self.parser = argparse.ArgumentParser()
      self.parser.add_argument('-f', '--filename', default=None, action='store', help='File name of context')  
      self.parser.add_argument('-w', '--topic', default=None, action='store', help='Topic from wikipedia')  
      self.parser.add_argument('-ta', '--temperature_answer', default=0.5, type=float, action='store', help='Temperature setting for answer generation')
      self.parser.add_argument('-tq', '--temperature_question', default=0.5, type=float,action='store', help='Temperature setting for question generation')
      self.parser.add_argument('-tw', '--temperature_wrong_answer', default=2.0, type=float,action='store', help='Temperature setting for wrong answers')
      self.parser.add_argument('-c', '--context_size', default=8, type=int, action='store', help='Number of sentences used for the context')  
      self.parser.add_argument('-q', '--max_questions', default=10, type=int,action='store', help='Number of questions to generate')  
      self.parser.add_argument('-a', '--answers', default=4, type=int,action='store', help='Number of answers including correct. Set to 0 to output only questions')  
      self.parser.add_argument('-sr', '--summarize_ratio', default=None, type=float, action='store', help='Summarization ratio (for example 0.2). Alternative to --summarize_word_count. Use 1.0 to disable summarization')  
      self.parser.add_argument('-sw', '--summarize_word_count', default=3000, type=int,action='store', help='Summarization word count (for example 3000). Alternative to --summarize_ratio')  
      self.parser.add_argument('-g', '--generate_count', default=20, type=int, action='store', help='Number of sequences generated each time. Higher values can produce better results but are slower and require more RAM')  
      self.parser.add_argument('-gs', '--generate_size', default=1, type=int, action='store', help='Number of attempts to generate sequences.')  
      self.parser.parse_args(namespace=self)



def preprocess(sentences):
    output = []
    for sent in sentences:
        single_quotes_present = len(re.findall(r"['][\w\s.:;,!?\\-]+[']",sent))>0
        double_quotes_present = len(re.findall(r'["][\w\s.:;,!?\\-]+["]',sent))>0
        question_present = "?" in sent
        quotes_present = '«' in sent or '»' in sent
        brackets_present = '(' in sent or ')' in sent
        if single_quotes_present or double_quotes_present or question_present or quotes_present or brackets_present:
            continue
        else:
            output.append(sent.strip(punctuation))
    return output
        
        

def clean_text(text):
  t = re.sub('===? ([^=]+) ===?', '\\1.', re.sub('\[[^]]{,10}\]', '', text))
  if t.find('Ссылки.\n') != -1:
    t = t[:t.find('Ссылки.\n')]
  if t.find('Примечания.\n') != -1:
    t = t[:t.find('Примечания.\n')]
  return t

def preprocess_mcq(sentences):
    output = []
    for sent in sentences:
        single_quotes_present = len(re.sub("[^']",'',sent)) > 1
        double_quotes_present = "\"" in sent
        question_present = "?" in sent
        quotes_present = '«' in sent or '»' in sent
        brackets_present = '(' in sent or ')' in sent
        toks = list(tokenize(sent))
        if len(list(toks)) == 0:
          continue
        found_det = False
        tok_texts = [x.text for x in list(toks)]
        #print(tok_texts)
        for index, token in enumerate(morph(tok_texts).tokens):
          if token.pos=='DET':
            #print(token)
            found_det = True
            break
        #single_quotes_present or double_quotes_present or quotes_present
        if question_present or brackets_present or found_det:
            continue
        else:
            output.append(sent.strip(punctuation))
    return output
        
        
def get_candidate_sents(resolved_text, words=None, ratio=0.2):
    if words is not None:
      candidate_sents = summarize(resolved_text, words=words, language='russian')
      candidate_sents_list = [x.text for x in list(sentenize(candidate_sents))]
    elif ratio is not None and ratio < 1:
      candidate_sents = summarize(resolved_text, ratio=ratio, language='russian')
      candidate_sents_list = [x.text for x in list(sentenize(candidate_sents))]
    else:
      candidate_sents_list = [x.text for x in list(sentenize(resolved_text))]
    #print('after summarize', len(candidate_sents_list))
    # candidate_sents_list = [re.split(r'[:;]+',x)[0] for x in candidate_sents_list ]
    # Remove very short sentences less than 30 characters and long sentences greater than 150 characters
    filtered_list_short_sentences = [sent for sent in candidate_sents_list if len(sent)>50 and len(sent)<250]
    return filtered_list_short_sentences
    

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
	

def add_br(s):
  i = 0
  r = ''
  while i < len(s):
    r += s[i:i+100] + "\n"
    i += 100
  if i < len(s):
    r += s[i:i+100]
  return r


def clean_answer(aa):
  return aa.strip(punctuation)
def similar_answer(a1, a2):
  a1 = a1.lower()
  a2 = a2.lower()
  if re.sub('[^\d]','',a1) == re.sub('[^\d]','',a2) and (re.sub('[^\d]','',a1) != '' or re.sub('[^\d]','',a2) != ''):
    return True
  if a1.find(a2) != -1 or a2.find(a1) != -1:
    return True
  if SequenceMatcher(None, a1, a2).ratio() > 0.5:
    return True
  return False
def join_answers(c):
  new_c = Counter()
  processed = []
  for i, ans in enumerate(c):
    if ans in processed:
      continue
    #print(ans)
    similar_ans = []
    for j, other in enumerate(c):
      if not other in processed and ans != other:
        if similar_answer(ans, other):
          similar_ans.append(other)
    max_similar_count = 0
    for other in similar_ans:
      max_similar_count = max(max_similar_count, c[other])
    if max_similar_count <= c[ans]:
      #print(ans, max_similar_count, c[ans])
      new_c[ans] = c[ans]
      for other in similar_ans:
        new_c[ans] += c[other]
        processed.append(other)
  return new_c




def word_set(s):
   return set([x.text for x in list(tokenize(s))])

def word_set_score(a,b):
   return len(a.intersection(b)) / len(a.union(b))
