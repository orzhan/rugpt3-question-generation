from razdel import tokenize, sentenize
from string import punctuation
import re
from random import shuffle
import random
import wikipedia
import argparse
import torch
import os
import argparse
import logging
import numpy as np
import wikipedia
from collections import Counter
from difflib import SequenceMatcher
from tools import MultipleChoiceArgs, clean_text, preprocess, get_candidate_sents, preprocess_mcq, set_seed, add_br, clean_answer, similar_answer, join_answers, word_set, word_set_score
import logging
lib_logger = logging.getLogger("russian-question-generation")


import gc


from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
)


def adjust_length_to_model(length, max_sequence_length):
    if length < 0 and max_sequence_length > 0:
        length = max_sequence_length
    elif 0 < max_sequence_length < length:
        length = max_sequence_length  # No generation bigger than model size
    elif length < 0:
        length = 10000  # avoid infinite loop
    return length


def init_model(model_name_or_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(1)

    tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path)
    model = GPT2LMHeadModel.from_pretrained(model_name_or_path)
    model.to(device)
    return (tokenizer, model)

def generate(prompt, temperature, length, num_return_sequences, p, tokenizer, model):
    length = adjust_length_to_model(length, max_sequence_length=model.config.max_position_embeddings)
    #logger.info(args)
    generated_sequences = []
    prompt_text = ""
    prompt_text = prompt 

    encoded_prompt = tokenizer.encode(prompt_text, add_special_tokens=False, return_tensors="pt")
    encoded_prompt = encoded_prompt.to(model.device)

    output_sequences = model.generate(
        input_ids=encoded_prompt,
        max_length=length + len(encoded_prompt[0]),
        temperature=temperature,
        top_k=0,
        top_p=p,
        #repetition_penalty=args.repetition_penalty,
        do_sample=True,
        num_return_sequences=num_return_sequences,
    )

    if len(output_sequences.shape) > 2:
        output_sequences.squeeze_()

    for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
        #print("ruGPT:".format(generated_sequence_idx + 1))
        generated_sequence = generated_sequence.tolist()

        # Decode text
        text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)

        # Remove all text after the stop token
        #text = text[: text.find(args.stop_token) if args.stop_token else None]

        # Add the prompt at the beginning of the sequence. Remove the excess text that was used for pre-processing
        total_sequence = (
            prompt_text + text[len(tokenizer.decode(encoded_prompt[0], clean_up_tokenization_spaces=True)) :]
        )

        generated_sequences.append(total_sequence)
     
    return generated_sequences

   
def generate_multiple_choice(args):
    if args.topic is not None:
        wikipedia.set_lang("ru")
        p = wikipedia.page(args.topic)
        text = p.content
    elif args.filename is not None:
        with open(args.filename, encoding='utf-8') as fin:
            text = fin.read()
    else:
        raise Exception("Topic or filename parameter is required")

    lib_logger.info('Loaded text')
    cleaned_text = clean_text(text)
    cand_sents = get_candidate_sents(cleaned_text, args.summarize_word_count, args.summarize_ratio)
    filter_quotes_and_questions = preprocess_mcq(cand_sents)
    lib_logger.info('Finished preprocessing text')

    random.shuffle(filter_quotes_and_questions)
    filter_quotes_and_questions = filter_quotes_and_questions[:args.max_questions * 2]

    lib_logger.info('Contexts initialized')
    tokenizer, model = init_model('question_model_large2')
    lib_logger.info('Loaded model')

    model_qa = None
    aqs_store = {}

    for sent_index,_ in enumerate(filter_quotes_and_questions):
        true = filter_quotes_and_questions[sent_index]
        toks = list(tokenize(true))
        sent_in_text = cleaned_text.find(true)
        context_sents = list(sentenize(cleaned_text[:sent_in_text]))[-args.context_size:]
        if len(context_sents) < args.context_size - 1:
          continue
        context = ' '.join([x.text for x in context_sents])
        aqs = generate('Context: ' + context + ' Answer: ', args.temperature_answer, 50, args.generate_count, 0.95, tokenizer, model)
        aqs_store[sent_index] = aqs

    lib_logger.info('Generated questions')
    if args.answers > 0:
        if 'model' in locals():
          del tokenizer
          torch.cuda.empty_cache()
          torch.cuda.synchronize()
          gc.collect()
        tokenizer, model_qa = init_model('answer_model_large')
        lib_logger.info('Loaded answering model')

    questions = []
    answers = []
    n_include = 0
    n = 0
    n_error = 0

    for sent_index,_ in enumerate(filter_quotes_and_questions):
        true = filter_quotes_and_questions[sent_index]
        toks = list(tokenize(true))
        sent_in_text = cleaned_text.find(true)
        context_sents = list(sentenize(cleaned_text[:sent_in_text]))[-args.context_size:]
        if len(context_sents) < args.context_size - 1:
          continue
        context = ' '.join([x.text for x in context_sents])
        aqs = aqs_store[sent_index]
        found = False
        for aq in aqs:
          try:
            #a = aq[aq.index('Answer: ')+len('Answer: '):].strip()
            q = aq[aq.index('Question: ')+len('Question: '):].strip()
            #a = a[:a.index('Question: ')].strip()
            if (q.find('Context: ') != -1):
              q = q[:q.find('Context: ')].strip()
            #and word_set_score(word_set(context), word_set(q)) >= 0.25
            if args.answers == 0:
                questions.append({'context':context,'question':q})
                break
            if len(q) > 25 and q.index('?') != -1:
              # test if model can answer the question itself
              qqa = generate('Context: ' + context + ' Question: ' + q + ' Answer: ', args.temperature_question, 30, args.generate_count, 0.95, tokenizer, model_qa)
              
              qqaw = generate('Question: ' + q + ' Answer: ', args.temperature_wrong_answer, 30, args.generate_count, 0.95, tokenizer, model_qa)
              
              aac = 0

              list_good = []
              list_bad = []
              for qa in qqa:
                aa = qa[qa.index('Answer: ')+len('Answer: '):].strip()
                if (aa.find('Context') != -1):
                  aa = aa[:aa.find('Context')].strip()
                list_good.append(clean_answer(aa))
              for qa in qqaw:
                aa = qa[qa.index('Answer: ')+len('Answer: '):].strip()
                if (aa.find('Context') != -1):
                  aa = aa[:aa.find('Context')].strip()
                if ('[' in aa) or (']' in aa) or ('(' in aa) or (')' in aa) or len(aa) >= 40 or len(aa) < 3:
                    continue
                list_bad.append(clean_answer(aa))

              c_good = Counter(list_good)
              c_bad = Counter(list_bad)
              c_good = join_answers(c_good)
              c_bad = join_answers(c_bad)
              wrong = [x[0] for x in c_bad.most_common(args.answers - 1)]
              aa = c_good.most_common(1)[0][0]
              if word_set_score(word_set(aa), word_set(q)) < 0.5 and q.find(aa) == -1 and len(aa) > 2 and len(aa) < 120:
                if c_good[aa]/len(list_good) > 0.35:
                  if len(wrong) >= 2:
                    a_context_intersect_score = 40 * min(0.025, word_set_score(word_set(aa), word_set(context)))
                    if a_context_intersect_score > 0:
                        rating = c_good[aa]/len(list_good) + a_context_intersect_score
                        questions.append({'context':context,'question':q,'answer':aa,'wrong':wrong,'rating':rating})
                        n_include += 1
                        found = True
                        break     
          except Exception as e:
            n_error += 1
            print(e)
        n += 1
        if n >= args.max_questions:
            break
        #if n % 10 == 0:
        #  print(f"{n}/{len(filter_quotes_and_questions)}")
    lib_logger.info('Finished generating answers')
    return questions
    
    

def main():
    print(generate_multiple_choice(MultipleChoiceArgs()))

if __name__ == "__main__":
    main()