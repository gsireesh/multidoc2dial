import itertools
import pickle
import os
import sys

from fire import Fire
from keybert import KeyBERT
import networkx as nx
import numpy as np
from tqdm import tqdm

sys.path.append("../../GRADE")
from preprocess.prepare_data import PreprocessTool
PATH_TEMPLATE = "/usr0/home/sgururaj/src/11-797-multidoc2dial/multidoc2dial/data/mdd_all/dd-generation-structure/{split}.{type}"

def get_source_file_lines(split):
    with open(PATH_TEMPLATE.format(split=split, type="source")) as f:
        return f.readlines()

def get_target_filename(split):
    return PATH_TEMPLATE.format(split=split, type="keywords")

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

def conceptify(word):
    return word.lower().replace(" ", "_")

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

    filtered_keywords = [
        [keyword[0] for keyword in turn
        if conceptify(keyword[0]) in c2id and c2id[conceptify(keyword[0])] in concept_graph.nodes()] 
        for turn in keywords_by_turn
    ]
    return filtered_keywords

def get_keywords_from_line(keyword_extractor, line, c2id, concept_graph):
    dialogue_utterances = clean_line(line)
    keywords_by_turn = keywordify_dialogue(keyword_extractor, dialogue_utterances, c2id, concept_graph)
    return keywords_by_turn

def construct_adjacency_matrices(keywords, c2id, concept_graph):
    all_keywords = list(itertools.chain(*keywords))
    adj_matrix = np.zeros(shape=(len(all_keywords), len(all_keywords)))
    for i, source_keyword in enumerate(all_keywords):
        s_id = c2id[conceptify(source_keyword)]
        for j, tgt_keyword in enumerate(all_keywords):
            t_id = c2id[conceptify(tgt_keyword)]
            try:
                shortest_path_length = nx.shortest_path_length(concept_graph, s_id, t_id)
            except nx.NetworkXNoPath:
                shortest_path_length = 0
            adj_matrix[i, j] = 1 / (shortest_path_length + 1)
    return adj_matrix


    

def generate_keywords_file(split):
    lines = get_source_file_lines(split)
    target_file_path = get_target_filename(split)
    
    keybert = KeyBERT()
    pt = PreprocessTool()
    c2id, _, _, _, = pt.load_resources()
    cpnet = pt.load_cpnet()


    print("extracting keywords...")
    keyword_lines = [get_keywords_from_line(keybert, line, c2id, cpnet) for line in tqdm(lines)]
    print("getting adjacency matrices")
    adjacency_matrices = [construct_adjacency_matrices(keyword_line, c2id, cpnet) for keyword_line in tqdm(keyword_lines)]


    with open(target_file_path, "wb") as f:
        pickle.dump((keyword_lines, adjacency_matrices), f)


if __name__ == "__main__":
    Fire(generate_keywords_file)