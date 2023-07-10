# Run QBERT on SQuAD

import json
import logging
import os
import random
import sys
import numpy as np
import torch
import pickle

from io import open
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from tqdm import tqdm, trange
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME
from modeling import BertForQuestionAnswering_Quant as BertForQuestionAnswering
from pytorch_pretrained_bert.optimization import BertAdam 
from pytorch_pretrained_bert.tokenization import (BertTokenizer, whitespace_tokenize)
from custom_utils import get_parser, InputFeatures

logger = logging.getLogger(__name__)

import warnings
warnings.filterwarnings("ignore")

class SquadExample(object):
    """
    A single training/test example for the Squad dataset.
    For examples without an answer, the start and end position are -1.
    """

    def __init__(self,
                 qas_id,
                 question_text,
                 doc_tokens,
                 orig_answer_text=None,
                 start_position=None,
                 end_position=None,
                 is_impossible=None):
        self.qas_id = qas_id
        self.question_text = question_text
        self.doc_tokens = doc_tokens
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "qas_id: %s" % (self.qas_id)
        s += ", question_text: %s" % (self.question_text)
        s += ", doc_tokens: [%s]" % (" ".join(self.doc_tokens))
        if self.start_position:
            s += ", start_position: %d" % (self.start_position)
        if self.end_position:
            s += ", end_position: %d" % (self.end_position)
        if self.is_impossible:
            s += ", is_impossible: %r" % (self.is_impossible)
        return s



def read_squad_examples(input_file, is_training, version_2_with_negative):
    """Read a SQuAD json file into a list of SquadExample."""
    with open(input_file, "r", encoding='utf-8') as reader:
        input_data = json.load(reader)["data"][0:10]

    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    examples = []
    for entry in input_data:
        for paragraph in entry["paragraphs"]:
            paragraph_text = paragraph["context"]
            doc_tokens = []
            char_to_word_offset = []
            prev_is_whitespace = True
            for c in paragraph_text:
                if is_whitespace(c):
                    prev_is_whitespace = True
                else:
                    if prev_is_whitespace:
                        doc_tokens.append(c)
                    else:
                        doc_tokens[-1] += c
                    prev_is_whitespace = False
                char_to_word_offset.append(len(doc_tokens) - 1)

            for qa in paragraph["qas"]:
                qas_id = qa["id"]
                question_text = qa["question"]
                start_position = None
                end_position = None
                orig_answer_text = None
                is_impossible = False
                if is_training:
                    if version_2_with_negative:
                        is_impossible = qa["is_impossible"]
                    if (len(qa["answers"]) != 1) and (not is_impossible):
                        raise ValueError(
                            "For training, each question should have exactly 1 answer."
                        )
                    if not is_impossible:
                        answer = qa["answers"][0]
                        orig_answer_text = answer["text"]
                        answer_offset = answer["answer_start"]
                        answer_length = len(orig_answer_text)
                        start_position = char_to_word_offset[answer_offset]
                        end_position = char_to_word_offset[answer_offset +
                                                           answer_length - 1]
                        # Only add answers where the text can be exactly recovered from the
                        # document. If this CAN'T happen it's likely due to weird Unicode
                        # stuff so we will just skip the example.
                        #
                        # Note that this means for training mode, every example is NOT
                        # guaranteed to be preserved.
                        actual_text = " ".join(
                            doc_tokens[start_position:(end_position + 1)])
                        cleaned_answer_text = " ".join(
                            whitespace_tokenize(orig_answer_text))
                        if actual_text.find(cleaned_answer_text) == -1:
                            continue
                    else:
                        start_position = -1
                        end_position = -1
                        orig_answer_text = ""

                example = SquadExample(
                    qas_id=qas_id,
                    question_text=question_text,
                    doc_tokens=doc_tokens,
                    orig_answer_text=orig_answer_text,
                    start_position=start_position,
                    end_position=end_position,
                    is_impossible=is_impossible)
                examples.append(example)
   
    return examples[0:120]
    #return examples



def run_squad_w_args(args):

    # select device to train the qbert model
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()
    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    # set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    tokenizer = BertTokenizer.from_pretrained(
        args.bert_model, do_lower_case=args.do_lower_case)

    train_examples = None
    num_train_optimization_steps = None
    if args.do_train:
        train_examples = read_squad_examples(
            input_file=args.train_file,
            is_training=True,
            version_2_with_negative=args.version_2_with_negative)
        num_train_optimization_steps = int(
            len(train_examples) / args.train_batch_size /
            args.gradient_accumulation_steps) * args.num_train_epochs

    # Prepare model
    cache_dir = os.path.join(
        str(PYTORCH_PRETRAINED_BERT_CACHE),
        'distributed_{}'.format(args.local_rank))
    model = BertForQuestionAnswering.from_pretrained(
        args.bert_model,
        cache_dir=cache_dir,
        config_dir=args.config_dir,
        config=args.config)

    model.to(device)

    # Prepare optimizer
    if args.do_train:
        param_optimizer = list(model.named_parameters())

        # hack to remove pooler, which is not used
        # thus it produce None grad that break apex
        param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [{
                                            'params': [
                                                p for n, p in param_optimizer
                                                if not any(nd in n for nd in no_decay)
                                            ],
                                            'weight_decay':
                                            0.01
                                        },
                                        {
                                            'params': [
                                                p for n, p in param_optimizer
                                                if any(
                                                    nd in n for nd in no_decay)
                                            ],
                                            'weight_decay':
                                            0.0
                                        }]

        optimizer = BertAdam(
                optimizer_grouped_parameters,
                lr=args.learning_rate,
                warmup=args.warmup_proportion,
                t_total=num_train_optimization_steps)

    global_step = 0
    total_loss = 0

    # start training
    if args.do_train:
        cached_train_features_file = args.train_file + '_{0}_{1}_{2}_{3}'.format(
            list(filter(None, args.bert_model.split('/'))).pop(),
            str(args.max_seq_length), str(args.doc_stride),
            str(args.max_query_length))
        train_features = None
        with open(cached_train_features_file, "rb") as reader:
            train_features = pickle.load(reader)
        #print("train_feautres", train_features)
        
        train_features=train_features[0:120]
        # train_features=train_features
        
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_start_positions = torch.tensor([f.start_position for f in train_features], dtype=torch.long)
        all_end_positions = torch.tensor([f.end_position for f in train_features], dtype=torch.long)
        
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_start_positions, all_end_positions)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(
            train_data,
            sampler=train_sampler,
            batch_size=args.train_batch_size)

        model.train()
        
        for _ in trange(int(args.num_train_epochs), desc="Epoch"):
            total_loss = 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, start_positions, end_positions = batch
                
                # forward
                loss = model(input_ids, segment_ids, input_mask, start_positions, end_positions)
                
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                # backward
                loss.backward()
                total_loss += loss.item()

                # optimization step (BERTAdam)
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1
                
                prof.step()


import torch
import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity

prof = torch.profiler.profile(
    schedule=torch.profiler.schedule(wait=1, warmup=0, active=3, repeat=1),
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/qbert'),
    record_shapes=True,
    with_stack=True)

def main():
    parser = get_parser()
    args = parser.parse_args()
    args.config = None
    
    prof.start()
    run_squad_w_args(args)
    prof.stop()
    
    print(prof.key_averages(group_by_stack_n=5).table(sort_by="self_cuda_time_total", row_limit=5))


if __name__ == "__main__":
    main()
