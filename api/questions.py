from razdel import tokenize, sentenize
from string import punctuation
import re
from random import shuffle
import random
import numpy as np
from collections import Counter
from difflib import SequenceMatcher
from tools import MultipleChoiceArgs, clean_text, preprocess, get_candidate_sents, preprocess_mcq, set_seed, add_br, clean_answer, similar_answer, join_answers, word_set, word_set_score
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from onnxruntime.transformers.gpt2_helper import Gpt2Helper, MyGPT2LMHeadModel
from transformers import AutoConfig
import onnxruntime
import numpy
import logging
from slovnet import Morph
from navec import Navec
from slovnet import NER


lib_logger = logging.getLogger("russian-question-generation")

morph = Morph.load('slovnet_morph_news_v1.tar', batch_size=4)
navec = Navec.load('navec_news_v1_1B_250K_300d_100q.tar')
morph.navec(navec)


model_name_or_path = '.'
cache_dir= '.'
config = AutoConfig.from_pretrained(model_name_or_path, cache_dir=".")
device = torch.device("cpu")

num_attention_heads = config.n_head
hidden_size = config.n_embd
num_layer = config.n_layer

onnx_model_path = "qamodel.onnx"
session = onnxruntime.InferenceSession(onnx_model_path)

lib_logger.info('Model is ready')

def get_tokenizer(model_name_or_path, cache_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, cache_dir=cache_dir)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = 0
    #okenizer.add_special_tokens({'pad_token': '[PAD]'})
    return tokenizer
    
tokenizer = get_tokenizer(model_name_or_path, cache_dir)

def get_example_inputs(prompt_text=''):    
    encodings_dict = tokenizer.batch_encode_plus(prompt_text, padding=True)

    input_ids = torch.tensor(encodings_dict['input_ids'], dtype=torch.int64)
    attention_mask = torch.tensor(encodings_dict['attention_mask'], dtype=torch.float32)
    position_ids = (attention_mask.long().cumsum(-1) - 1)
    position_ids.masked_fill_(position_ids < 0, 0)

    #Empty Past State for generating first word
    empty_past = []
    batch_size = input_ids.size(0)
    sequence_length = input_ids.size(1)
    past_shape = [2, batch_size, num_attention_heads, 0, hidden_size // num_attention_heads]
    for i in range(num_layer):
        empty_past.append(torch.empty(past_shape).type(torch.float32).to(device))
       
    return input_ids, attention_mask, position_ids, empty_past


def test_generation(tokenizer, input_text, ort_session=None, num_tokens_to_produce = 30, p=0.9, k=30):
    use_onnxruntime = (ort_session is not None)
    print("Text generation using", "OnnxRuntime" if use_onnxruntime else "PyTorch", "...")
    eos_token_id = 50256
    
    input_ids, attention_mask, position_ids, past = get_example_inputs(input_text)
    batch_size = input_ids.size(0)

    has_eos = torch.zeros(batch_size, dtype=torch.bool)

    all_token_ids = torch.tensor([]) #input_ids.clone()

    for step in range(num_tokens_to_produce):
        if ort_session is not None:
            outputs = inference_with_io_binding(ort_session, config, input_ids, position_ids, attention_mask, past)

        next_token_logits = outputs[0][:, -1, :]
        #print('next_token_logits', next_token_logits.shape)
        # Greedy approach is used here. You can easily extend it to use beam search and sampling to pick next tokens.


        # top_p sampling
        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
        cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
        sorted_indices_to_remove = cumulative_probs > p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
        next_token_logits = next_token_logits.masked_fill(indices_to_remove, 0)
        # top_k
        indices_to_remove = next_token_logits < torch.topk(next_token_logits, k)[0][..., -1, None]
        next_token_logits = next_token_logits.masked_fill(indices_to_remove, 0)
        
        
        #next_tokens = torch.argmax(next_token_logits, dim=-1)
        probs = F.softmax(next_token_logits, dim=-1)
        next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
        
        #print('next_tokens', next_tokens.shape)


        has_eos = has_eos | (next_tokens == eos_token_id)
        tokens_to_add = next_tokens.masked_fill(has_eos, eos_token_id)
        all_token_ids = torch.cat([all_token_ids, tokens_to_add.unsqueeze(-1)], dim=-1)
        
        #print('all_token_ids', all_token_ids.shape)

        # Update input_ids, attention_mask, position_ids and past
        input_ids = tokens_to_add.clone().detach().reshape([batch_size, 1]).to(device)    
        #print('input_ids', input_ids.shape)
        position_ids = (position_ids[:,-1] + 1).reshape(batch_size,1)
        #print('position_ids', position_ids.shape)
        attention_mask = torch.cat([attention_mask, torch.ones([batch_size, 1]).type_as(attention_mask)], 1).to(device)    
        #print('attention_mask', attention_mask.shape)

        past = []
        if not use_onnxruntime:
            past = list(outputs[1]) # past in torch output is tuple
        else:
            for i in range(num_layer):
                past_i = torch.from_numpy(outputs[i + 1]) if isinstance(outputs[i + 1], numpy.ndarray) else outputs[i + 1].clone().detach()
                past.append(past_i.to(device))

        if torch.all(has_eos):
            break

    result = []
    for i, output in enumerate(all_token_ids):
        #print("------------")
        #print(output)
        #print(tokenizer.decode([int(x) for x in output], skip_special_tokens=True))
        result.append(tokenizer.decode([int(x) for x in output], skip_special_tokens=True))
        
    return result
        
def inference_with_io_binding(session, config, input_ids, position_ids, attention_mask, past):
    output_shapes = Gpt2Helper.get_output_shapes(batch_size=input_ids.size(0),
                                                 past_sequence_length=past[0].size(3),
                                                 sequence_length=input_ids.size(1),
                                                 config=config)
    output_buffers = Gpt2Helper.get_output_buffers(output_shapes, device)

    io_binding = Gpt2Helper.prepare_io_binding(session, input_ids, position_ids, attention_mask, past,
                                               output_buffers, output_shapes)
    session.run_with_iobinding(io_binding)

    outputs = Gpt2Helper.get_outputs_from_io_binding_buffer(session, output_buffers, output_shapes,
                                                            return_numpy=False)
    return outputs
    

def generate(prompt, temperature, length, num_return_sequences, p=0.9, k=30):
    return test_generation(tokenizer, [prompt] * num_return_sequences, session, length, p, k)

def generate_multiple_choice(text, max_questions = 3, max_wrong = 4):
    lib_logger.info('Loaded text')
    cleaned_text = clean_text(text)
    cand_sents = get_candidate_sents(cleaned_text, 3000, None)
    context_size = min(len(cand_sents)//2, 5)
    filter_quotes_and_questions = preprocess_mcq(cand_sents)
    lib_logger.info('Finished preprocessing text, sentence count %d' % len(filter_quotes_and_questions))

    random.shuffle(filter_quotes_and_questions)
    filter_quotes_and_questions = filter_quotes_and_questions[:max_questions * 2]
    if len(filter_quotes_and_questions) < 2:
        filter_quotes_and_questions = [text]
        context_size = 0
    if len(text) < 10:
        return {'error': 'text too short'}
    
    lib_logger.info('Contexts initialized')
    #tokenizer, model = init_model('question_model_large2')
    lib_logger.info('Loaded model')

    
    questions = []
    

    for sent_index,_ in enumerate(filter_quotes_and_questions):
        true = filter_quotes_and_questions[sent_index]
        toks = list(tokenize(true))
        sent_in_text = cleaned_text.find(true)
        if context_size > 0:
            context_sents = list(sentenize(cleaned_text[:sent_in_text]))[-context_size:]
            if len(context_sents) < context_size - 1:
                continue
            context = ' '.join([x.text for x in context_sents])
            if len(context) == 0:
                continue
        else:
            context = true
        aqs = generate('Context: ' + context + ' Question: ', 0.7, 50, 3, 0.9, 1)
        for q in aqs:
            if (q.find('Context: ') != -1):
              q = q[:q.find('Context: ')].strip()
            a = ''
            if (q.find('Answer') != -1):
              a = q[q.find('Answer')+len('Answer')+1:].strip()
              q = q[:q.find('Answer')].strip()
            print('generated q', q)
            print('generated a', a)
            was_good_question = False
            if len(q) > 0 and q.index('?') != -1:
                if len(a) > 0 and len(a) < 35 and context.find(a) != -1:
                    found = False
                    for qq in questions:
                        if qq['question'] == q:
                            found = True
                            break
                    if not found:
                        if max_wrong > 0:
                            rc = random.randint(0, context_size - 1)
                            context_wrong = 'Context: ' + ' '.join([x.text for x in context_sents if x.text.find(a)==-1][rc:rc+1])
                            aaa = generate(context_wrong + ' Question: ' + q + ' Answer: ', 1.5, 12, max_wrong + 5, 0.9, 50)
                            print('generated wrong',aaa)
                            for j,wa in enumerate(aaa):
                                if wa.find('\n') != -1:
                                    wa = wa[:wa.find('\n')]
                                if wa.find('.') != -1:
                                    wa = wa[:wa.find('.')]
                                aaa[j] = wa.strip()
                            print('cut wrong', aaa)
                            aaa = [wa for wa in aaa if len(wa) > 2 and len(wa) < 35 and context.find(wa) == -1 and wa.find(a) == -1 and a.find(wa) == -1]
                            print('filtered wrong',aaa)
                        else:
                            aaa = []
                        if max_wrong == 0 or len(aaa) >= 2:
                            questions.append({'context':context,'question':q,'answer':a,'wrong':aaa[:max_wrong]})
                            lib_logger.info('Done: %d of %d' % (len(questions), max_questions))
                            was_good_question = True
                        if len(questions) >= max_questions:
                            break
            if was_good_question:
                break
        if len(questions) >= max_questions:
            break
        
        

    lib_logger.info('Generated questions')
    return questions
    
   
def generate_true_false(text, max_questions=3):
    lib_logger.info('Loaded text')
    cleaned_text = clean_text(text)
    cand_sents = get_candidate_sents(cleaned_text, 3000, None)
    context_size = min(len(cand_sents)//2, 5)
    filter_quotes_and_questions = preprocess(cand_sents)
    lib_logger.info('Finished preprocessing text, sentence count %d' % len(filter_quotes_and_questions))

    random.shuffle(filter_quotes_and_questions)
    filter_quotes_and_questions = filter_quotes_and_questions[:max_questions * 2]
    if len(filter_quotes_and_questions) < 2:
        filter_quotes_and_questions = [text]
        context_size = 0
    if len(text) < 10:
        return {'error': 'text too short'}
            
    questions = []
    answers=[]
    
    print('go')

    for sent_index,_ in enumerate(filter_quotes_and_questions):
      #try:
        true = filter_quotes_and_questions[sent_index]
        print('true', true)
        toks = list(tokenize(true))
        sent_in_text = cleaned_text.find(true)
        context = ' '.join([x.text for x in list(sentenize(cleaned_text[:sent_in_text]))[-context_size:]])
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
            print('skip')
            continue
          noun_near_half = random.choice(noun_index)
          if noun_near_half in noun_near_halfs:
            print('noun_near_half')
            continue
          noun_near_halfs.append(noun_near_half)
          #print(noun_near_half)
          sent_part = true[:toks[noun_near_half-1].stop]
          #generated = generate_bs(context + '. ' + sent_part,50,0.9)
          print('sent_part', sent_part)
          generated = []
          #for j in range(0,5):
          generated += generate(context + '. ' + sent_part, 0.7, 50, 10)
          print('generated', generated)
          true_len = len(true)-len(sent_part)
          #false = [list(sentenize(s))[context_size].text for s in generated]
          false = generated
          #false = [x for x in false if len(x) >= true_len // 2 and len(x) <= true_len * 2][-5:]
          #false = [x for x in false if sentence_similarity(x, true) < 0.99]
          #print('similarity',similar,true,false)
          #if similar < 0.7:
          for j, f in enumerate(false):
            if f.find('.') > 2:
                false[j] = f[:f.find('.')]
            print(false[j])
          false = [x for x in false if len(x) >= true_len // 3 and len(x) <= true_len * 3][-5:]
          if len(false) > 0:
            falses.extend(false)
            print('falses', falses)
            break
        if len(falses) == 0:
          print('empty falses')
          continue
        questions.append(sent_part)
        answers.append([true[len(sent_part):]] + [falses]) 
        print('generate', questions[-1], answers[-1])
        #false = random.choice(falses)
        #v = random.choice([True,False])
        #if not v:
        #  questions.append(false)
        #else:
        #  questions.append(true)
        #correct_answers.append(v)
        if (len(questions) >= max_questions):
          break
      #except Exception as e:
      #  print(e)
      #  pass

    print('result of generation', list(zip(questions, answers)))
    return list(zip(questions, answers))
#print(generate_multiple_choice("""""", 3))
#with open('input.txt', encoding='utf-8') as f:
#    print(generate_multiple_choice(f.read(), 3))