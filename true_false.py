from summa.summarizer import summarize
from razdel import tokenize, sentenize
from slovnet import Morph
from navec import Navec
from slovnet import NER
from string import punctuation
import re
from random import shuffle
import spacy
import random
import wikipedia
import argparse
from tools import TrueFalseArgs, clean_text, preprocess, get_candidate_sents

morph = Morph.load('slovnet_morph_news_v1.tar', batch_size=4)
navec = Navec.load('navec_news_v1_1B_250K_300d_100q.tar')
morph.navec(navec)

import torch
from transformers import AutoTokenizer, AutoModel, AutoModelWithLMHead, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoModelForMaskedLM
model_name_or_path = "sberbank-ai/rugpt3large_based_on_gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
model.eval()
model.to('cuda')


def generate_bs(prompt_text, size = 100, t=1):
  # Different models need different input formatting and/or extra arguments
  prefix = ""
  encoded_prompt = tokenizer.encode(prefix + prompt_text, add_special_tokens=False, return_tensors="pt")
  encoded_prompt = encoded_prompt.to('cuda')

  if encoded_prompt.size()[-1] == 0:
      input_ids = None
  else:
      input_ids = encoded_prompt

  output_sequences = model.generate(
        input_ids=input_ids,
        max_length=size + len(encoded_prompt[0]),
        temperature=t,
        top_k=0,
        top_p=0.95,
        repetition_penalty=1.0,
        do_sample=True,
        num_return_sequences=10,
    )
  if len(output_sequences.shape) > 2:
        output_sequences.squeeze_()

  generated_sequences = []

  for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
      generated_sequence = generated_sequence.tolist()

      text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)

      total_sequence = (
          prompt_text + text[len(tokenizer.decode(encoded_prompt[0], clean_up_tokenization_spaces=True)) :]
      )

      generated_sequences.append(total_sequence)

  return generated_sequences



def generate_true_false(args):
    if args.topic is not None:
        wikipedia.set_lang("ru")
        p = wikipedia.page(args.topic)
        text = p.content
    elif args.filename is not None:
        with open(args.filename, encoding='utf-8') as fin:
            text = fin.read()
    else:
        raise Exception("Topic or filename parameter is required")

            
    questions = []
    correct_answers=[]
    
    cleaned_text = clean_text(text)
    cand_sents = get_candidate_sents(cleaned_text, args.summarize_word_count, args.summarize_ratio)
    filter_quotes_and_questions = preprocess(cand_sents)

    random.shuffle(filter_quotes_and_questions)
    filter_quotes_and_questions = filter_quotes_and_questions[:args.max_questions*2]

    for sent_index,_ in enumerate(filter_quotes_and_questions):
      try:
        true = filter_quotes_and_questions[sent_index]
        toks = list(tokenize(true))
        sent_in_text = cleaned_text.find(true)
        context = ' '.join([x.text for x in list(sentenize(cleaned_text[:sent_in_text]))[-args.context_size:]])
        noun_near_half_dist = len(toks)
        falses = []
        noun_near_halfs = []
        skip = False
        for n_try in range(0,2):
          noun_near_half = -1
          noun_index = []
          for index, token in enumerate(morph([x.text for x in list(toks)]).tokens):
            if token.pos == 'PRON' and index < 4:
              skip = True
            if token.pos == 'DET' and index <= max(5, len(toks)/3):
              skip = True
            if token.pos=='NOUN' or token.pos=='VERB':
              dist = abs(index - len(toks) / 2)
              if index > 2 and index < len(toks)*0.67 + 1:
                noun_index.append(index)
              if dist < noun_near_half_dist:
                noun_near_half_dist = dist
                noun_near_half = index
          if skip:
            continue
          noun_near_half = random.choice(noun_index)
          if noun_near_half in noun_near_halfs:
            continue
          noun_near_halfs.append(noun_near_half)
          #print(noun_near_half)
          sent_part = true[:toks[noun_near_half-1].stop]
          generated = generate_bs(context + '. ' + sent_part,50,args.temperature)
          true_len = len(true)-len(sent_part)
          false = [list(sentenize(s))[args.context_size].text for s in generated]
          false_lens = [len(s)-len(sent_part) for s in false]
          false = [x for x in false if len(x)-len(sent_part) >= true_len and len(x)-len(sent_part) <= true_len * 2][-5:]
          #false = [x for x in false if sentence_similarity(x, true) < 0.99]
          #print('similarity',similar,true,false)
          #if similar < 0.7:
          falses.extend(false)
        if len(falses) == 0:
          continue
        false = random.choice(falses)
        v = random.choice([True,False])
        if not v:
          questions.append(false)
        else:
          questions.append(true)
        correct_answers.append(v)
        if (len(questions) >= args.max_questions):
          break
      except Exception as e:
        pass


    return list(zip(questions, correct_answers))
    
def main():
    print(generate_true_false(TrueFalseArgs()))

if __name__ == "__main__":
   main()