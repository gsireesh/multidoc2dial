from ast import keyword
from base64 import encode
import itertools
import json
from lib2to3.pgen2 import token
import linecache
import os
import pickle
import re
import socket
import string
import sys
from collections import Counter
from logging import getLogger
from pathlib import Path
from typing import Callable, Dict, Iterable, List
from sacrebleu import corpus_bleu

import git
import torch
from torch.utils.data import Dataset

from rank_bm25 import BM25Okapi
from datasets import load_dataset

from transformers import BartTokenizer, RagTokenizer, T5Tokenizer, BatchEncoding

sys.path.append("../../GRADE")
from preprocess.prepare_data import PreprocessTool
import networkx as nx
from keybert import KeyBERT

def load_bm25(in_path):
    dataset = load_dataset("csv", data_files=[in_path], split="train", delimiter="\t", column_names=["title", "text"])
    passages = []
    for ex in dataset:
        passages.extend(ex["text"].split("####"))
    passages_tokenized = [passage.strip().lower().split() for passage in passages]
    bm25 = BM25Okapi(passages_tokenized)
    return bm25


def get_top_n_indices(bm25, query, n=5):
    query = query.lower().split()
    scores = bm25.get_scores(query)
    scores_i = [(i, score) for i, score in enumerate(scores)]
    sorted_indices = sorted(scores_i, key=lambda score: score[1], reverse=True)
    return [x[0] for x in sorted_indices[:n]]


def load_bm25_results(in_path):
    d_query_pid = {}
    total = 0
    for split in ["train", "val", "test"]:
        queries, bm_rslt = [], []
        with open(os.path.join(in_path, f"{split}.source")) as f:
            for line in f:
                queries.append(line.strip())
        with open(os.path.join(in_path, f"{split}.bm25")) as f:
            for line in f:
                bm_rslt.append([int(ele) for ele in line.strip().split("\t")])
        total += len(queries)
        d_query_pid.update(dict(zip(queries, bm_rslt)))
    return d_query_pid

def keywordify_dialogue(keyword_extractor, dialogue, c2id, concept_graph):
    keywords_by_turn = [
        keyword_extractor.extract_keywords(
            turn,
            keyphrase_ngram_range=(1, 2), 
            stop_words='english', 
            use_mmr=True,
            diversity=0.7, 
            top_n=5
            ) 
        for turn in dialogue]

    filtered_keywords = [[keyword[0].replace(" ", "_") for keyword in turn if keyword[0].replace(" ", "_") in c2id and c2id[keyword[0].replace(" ", "_")] in concept_graph.nodes()] for turn in keywords_by_turn]
    return filtered_keywords

def clean_line(line):
    processed_dialogue = []
    dialogue = line.split("||")
    question, r1 = dialogue[0].split("[SEP]")
    for turn in [question, r1, *dialogue[1:]]:
        if turn.startswith("agent:"):
            processed_dialogue.append(turn[len("agent: "):])
        elif turn.startswith("user:"):
            processed_dialogue.append(turn[len("user: "):])
        else:
            processed_dialogue.append(turn)
    return processed_dialogue

def encode_keywords(tokenizer, keywords, keyword_turn_mask, max_length, padding_side, pad_to_max_length=True):
    
    extra_kw = {"add_prefix_space": True} if isinstance(tokenizer, BartTokenizer) else {}
    tokenizer.padding_side = padding_side
    if not keywords:
        keywords = ["placeholder"]
        keyword_turn_mask = [1]
    tokenizer_outs = tokenizer(
        keywords,
        truncation=True,
        padding=True, 
        return_tensors="pt",
        add_special_tokens=False,
        return_length=True,
        **extra_kw,
    )
    input_ids = [101]
    attention_mask = [1]
    token_type_ids  = [0]
    turn_mask = [-1]
    keyword_idx_map = [-1]


    for i, (keyword_token_list, turn_idx) in enumerate(zip(tokenizer_outs["input_ids"], keyword_turn_mask)):
        n_tokens = keyword_token_list.nonzero()[-1].item() + 1
        turn_mask.extend([turn_idx] * n_tokens)
        keyword_idx_map.extend([i] * n_tokens)
        input_ids.extend(keyword_token_list[:n_tokens])
        attention_mask.extend(tokenizer_outs["attention_mask"][i][:n_tokens])
        token_type_ids.extend(tokenizer_outs["token_type_ids"][i][:n_tokens])

    # add the EOS token
    input_ids.append(102)
    attention_mask.append(1)
    token_type_ids.append(0)
    turn_mask.append(-1)
    keyword_idx_map.append(-1)

    # not worrying about truncation here because this is just keywords, it really shouldn't be an issue
    if len(input_ids) > max_length:
        raise AssertionError("Somehow, you have more keywords than max_len.")

    if pad_to_max_length:
        input_ids.extend([0] * (max_length - len(input_ids)))
        attention_mask.extend([0] * (max_length - len(attention_mask)))
        token_type_ids.extend([0] * (max_length - len(token_type_ids)))
        turn_mask.extend([-1] * (max_length - len(turn_mask)))
        keyword_idx_map.extend([-1] * (max_length - len(keyword_idx_map)))

    return BatchEncoding({
        "input_ids": torch.tensor(input_ids, dtype=torch.int64),
        "token_type_ids": torch.tensor(token_type_ids, dtype=torch.int64),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.int64),
        "turn_mask": torch.tensor(turn_mask, dtype=torch.int64),
        "keyword_idx_map": torch.tensor(keyword_idx_map, dtype=torch.int64)
    })

    


def encode_line(tokenizer, line, max_length, padding_side, pad_to_max_length=True, return_tensors="pt"):
    extra_kw = {"add_prefix_space": True} if isinstance(tokenizer, BartTokenizer) and not line.startswith(" ") else {}
    tokenizer.padding_side = padding_side
    return tokenizer(
        [line],
        max_length=max_length,
        padding="max_length" if pad_to_max_length else None,
        truncation=True,
        return_tensors=return_tensors,
        add_special_tokens=True,
        **extra_kw,
    )


def encode_line2(tokenizer, line, max_length, padding_side, pad_to_max_length=True, return_tensors="pt"):
    extra_kw = {"add_prefix_space": True} if isinstance(tokenizer, BartTokenizer) and not line.startswith(" ") else {}
    tokenizer.padding_side = padding_side
    line = tuple(line.split("[SEP]"))
    return tokenizer(
        [line],
        max_length=max_length,
        padding="max_length" if pad_to_max_length else None,
        truncation=True,
        return_tensors=return_tensors,
        add_special_tokens=True,
        **extra_kw,
    )


def trim_batch(
    input_ids,
    pad_token_id,
    attention_mask=None,
):
    """Remove columns that are populated exclusively by pad_token_id"""
    keep_column_mask = input_ids.ne(pad_token_id).any(dim=0)
    if attention_mask is None:
        return input_ids[:, keep_column_mask]
    else:
        return (input_ids[:, keep_column_mask], attention_mask[:, keep_column_mask])


class Seq2SeqDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        data_dir,
        max_source_length,
        max_target_length,
        type_path="train",
        n_obs=None,
        src_lang=None,
        tgt_lang=None,
        prefix="",
    ):
        super().__init__()
        self.src_file = Path(data_dir).joinpath(type_path + ".source")
        self.tgt_file = Path(data_dir).joinpath(type_path + ".target")
        self.domain_file = Path(data_dir).joinpath(type_path + ".domain")
        if not os.path.exists(self.domain_file):
            self.domain_file = None
        self.src_lens = self.get_char_lens(self.src_file)
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        assert min(self.src_lens) > 0, f"found empty line in {self.src_file}"
        self.tokenizer = tokenizer
        self.prefix = prefix
        if n_obs is not None:
            self.src_lens = self.src_lens[:n_obs]
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        with open(os.path.join(os.getenv("DATA_DIR"), f"{type_path}.keywords"), "rb") as f:
            self.keyword_lines, self.adjacency_matrices = pickle.load(f)

        pt = PreprocessTool()
        self.c2id, _, _, _, = pt.load_resources()

    def __len__(self):
        return len(self.src_lens)

    def __getitem__(self, index) -> Dict[str, torch.Tensor]:
        index = index + 1  # linecache starts at 1
        source_line = self.prefix + linecache.getline(str(self.src_file), index).rstrip("\n")
        tgt_line = linecache.getline(str(self.tgt_file), index).rstrip("\n")

        assert source_line, f"empty source line for index {index}"
        assert tgt_line, f"empty tgt line for index {index}"
        domain_line = None
        if self.domain_file is not None:
            domain_line = linecache.getline(str(self.domain_file), index).rstrip("\n")
            assert domain_line, f"empty domain line for index {index}"

        # fixing index so that it corresponds to linecache
        keywords_by_turn = self.keyword_lines[index - 1]
        adjacency_matrix = torch.tensor(self.adjacency_matrices[index - 1])

        all_keywords = list(itertools.chain(*keywords_by_turn))
        turn_mask = [1] * len(keywords_by_turn[0]) + [0] * len(list(itertools.chain(*keywords_by_turn[1:])))
        all_keyword_ids = torch.tensor([self.c2id[keyword.replace(" ", "_")] for keyword in all_keywords], dtype=torch.int64)
    

        # Need to add eos token manually for T5
        if isinstance(self.tokenizer, T5Tokenizer):
            source_line += self.tokenizer.eos_token
            tgt_line += self.tokenizer.eos_token

        # Pad source and target to the right
        source_tokenizer = (
            self.tokenizer.question_encoder if isinstance(self.tokenizer, RagTokenizer) else self.tokenizer
        )
        target_tokenizer = self.tokenizer.generator if isinstance(self.tokenizer, RagTokenizer) else self.tokenizer
        
        keyword_inputs = encode_keywords(source_tokenizer, all_keywords, turn_mask, self.max_source_length, "right")
        target_inputs = encode_line(target_tokenizer, tgt_line, self.max_target_length, "right")

        source_ids = keyword_inputs["input_ids"]
        target_ids = target_inputs["input_ids"].squeeze()
        src_mask = keyword_inputs["attention_mask"]
        src_token_type_ids = keyword_inputs["token_type_ids"]
        return {
            "input_ids": source_ids,
            "attention_mask": src_mask,
            "token_type_ids": src_token_type_ids,
            "decoder_input_ids": target_ids,
            "domain": domain_line,
            "turn_mask": keyword_inputs["turn_mask"],
            "keyword_idx_map": keyword_inputs["keyword_idx_map"],
            "keyword_ids": all_keyword_ids,
            "adjacency_matrices": adjacency_matrix
        }

    @staticmethod
    def get_char_lens(data_file):
        return [len(x) for x in Path(data_file).open().readlines()]

    # TODO: make sure to modify this!!
    def collate_fn(self, batch) -> Dict[str, torch.Tensor]:
        input_ids = torch.stack([x["input_ids"] for x in batch])
        masks = torch.stack([x["attention_mask"] for x in batch])
        token_type_ids = torch.stack([x["token_type_ids"] for x in batch])
        target_ids = torch.stack([x["decoder_input_ids"] for x in batch])
        domain = [x["domain"] for x in batch]
        turn_masks = torch.stack([x["turn_mask"] for x in batch])
        keyword_idx_map = torch.stack([x["keyword_idx_map"] for x in batch])
        keyword_ids = torch.nn.utils.rnn.pad_sequence([x["keyword_ids"] for x in batch], batch_first=True, padding_value=-1)

        max_n_keywords = max([len(x["keyword_ids"]) for x in batch])
        adjacency_matrices_list = []
        for instance in batch:
            adj = instance["adjacency_matrices"]
            pad_size = max_n_keywords - adj.shape[0]
            adjacency_matrix = torch.nn.functional.pad(adj, (0, pad_size, 0, pad_size), value=-1)
            adjacency_matrices_list.append(adjacency_matrix)

        adjacency_matrices = torch.stack(adjacency_matrices_list)


        tgt_pad_token_id = (
            self.tokenizer.generator.pad_token_id
            if isinstance(self.tokenizer, RagTokenizer)
            else self.tokenizer.pad_token_id
        )
        src_pad_token_id = (
            self.tokenizer.question_encoder.pad_token_id
            if isinstance(self.tokenizer, RagTokenizer)
            else self.tokenizer.pad_token_id
        )

        y = trim_batch(target_ids, tgt_pad_token_id)
        source_ids, source_mask = trim_batch(input_ids, src_pad_token_id, attention_mask=masks)
        keep_col_mask = input_ids.ne(src_pad_token_id).any(dim=0)
        token_type_ids = token_type_ids[:, keep_col_mask]
        batch = {
            "input_ids": source_ids,
            "attention_mask": source_mask,
            "token_type_ids": token_type_ids,
            "decoder_input_ids": y,
            "domain": domain,
            "turn_mask": turn_masks,
            "keyword_idx_map": keyword_idx_map,
            "keyword_ids": keyword_ids,
            "adjacency_matrices": adjacency_matrices

        }
        return batch


logger = getLogger(__name__)


def flatten_list(summary_ids: List[List]):
    return [x for x in itertools.chain.from_iterable(summary_ids)]


def save_git_info(folder_path: str) -> None:
    """Save git information to output_dir/git_log.json"""
    repo_infos = get_git_info()
    save_json(repo_infos, os.path.join(folder_path, "git_log.json"))


def save_json(content, path, indent=4, **json_dump_kwargs):
    with open(path, "w") as f:
        json.dump(content, f, indent=indent, **json_dump_kwargs)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def get_git_info():
    repo = git.Repo(search_parent_directories=True)
    repo_infos = {
        "repo_id": str(repo),
        "repo_sha": str(repo.head.object.hexsha),
        "repo_branch": str(repo.active_branch),
        "hostname": str(socket.gethostname()),
    }
    return repo_infos


def lmap(f: Callable, x: Iterable) -> List:
    """list(map(f, x))"""
    return list(map(f, x))


def pickle_save(obj, path):
    """pickle.dump(obj, path)"""
    with open(path, "wb") as f:
        return pickle.dump(obj, f)


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def calculate_exact_match(output_lns: List[str], reference_lns: List[str]) -> Dict:
    assert len(output_lns) == len(reference_lns)
    em = 0
    for hypo, pred in zip(output_lns, reference_lns):
        em += exact_match_score(hypo, pred)
    if len(output_lns) > 0:
        em /= len(output_lns)
    return {"em": em}


def calculate_bleu(output_lns, refs_lns) -> dict:
    """Uses sacrebleu's corpus_bleu implementation."""
    return {"bleu": round(corpus_bleu(output_lns, [refs_lns]).score, 4)}


def is_rag_model(model_prefix):
    return model_prefix.startswith("rag")


def set_extra_model_params(extra_params, hparams, config):
    equivalent_param = {p: p for p in extra_params}
    # T5 models don't have `dropout` param, they have `dropout_rate` instead
    equivalent_param["dropout"] = "dropout_rate"
    for p in extra_params:
        if getattr(hparams, p, None):
            if not hasattr(config, p) and not hasattr(config, equivalent_param[p]):
                logger.info("config doesn't have a `{}` attribute".format(p))
                delattr(hparams, p)
                continue
            set_p = p if hasattr(config, p) else equivalent_param[p]
            setattr(config, set_p, getattr(hparams, p))
            delattr(hparams, p)
    return hparams, config
