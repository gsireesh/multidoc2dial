from modeling_rag_dialdoc import *

class DialDocRagTokenForGenerationWithRider(DialDocRagTokenForGeneration):

    def __init__(self, *args, **kwargs):
        super(*args, **kwargs)

    def generate(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        context_input_ids=None,
        context_attention_mask=None,
        doc_scores=None,
        domain=None,
        max_length=None,
        min_length=None,
        early_stopping=None,
        use_cache=None,
        num_beams=None,
        num_beam_groups=None,
        diversity_penalty=None,
        bos_token_id=None,
        pad_token_id=None,
        eos_token_id=None,
        length_penalty=None,
        no_repeat_ngram_size=None,
        encoder_no_repeat_ngram_size=None,
        repetition_penalty=None,
        bad_words_ids=None,
        num_return_sequences=None,
        decoder_start_token_id=None,
        n_docs=None,
        prefix_allowed_tokens_fn: Callable[[int, torch.Tensor], List[int]] = None,
        forced_bos_token_id: Optional[int] = None,
        forced_eos_token_id: Optional[int] = None,
        remove_invalid_values: Optional[bool] = None,
        **model_kwargs,
    ):

    # set default parameters
        n_docs = n_docs if n_docs is not None else self.config.n_docs
        num_beams = num_beams if num_beams is not None else self.config.num_beams
        num_beam_groups = num_beam_groups if num_beam_groups is not None else self.config.num_beam_groups
        max_length = max_length if max_length is not None else self.config.max_length
        num_return_sequences = (
            num_return_sequences if num_return_sequences is not None else self.config.num_return_sequences
        )
        bos_token_id = bos_token_id if bos_token_id is not None else self.config.generator.bos_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.generator.eos_token_id
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.generator.pad_token_id
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        decoder_start_token_id = (
            decoder_start_token_id
            if decoder_start_token_id is not None
            else self.config.generator.decoder_start_token_id
        )
        remove_invalid_values = (
            remove_invalid_values if remove_invalid_values is not None else self.config.remove_invalid_values
        )

     # retrieve docs
        dialog_lengths = None
        if self.retriever is not None and context_input_ids is None:
            if self.config.scoring_func != "original":
                dpr_out = self.question_encoder(
                    input_ids, attention_mask=attention_mask, output_hidden_states=True, return_dict=True
                )
                combined_out = dpr_out.pooler_output

                ## Get mask for current turn input ids
                curr_turn_mask = torch.logical_xor(attention_mask, token_type_ids)
                current_turn_input_ids = input_ids * curr_turn_mask
                current_turn_only_out = self.question_encoder(
                    current_turn_input_ids, attention_mask=curr_turn_mask.long(), return_dict=True
                )
                current_turn_output = current_turn_only_out.pooler_output

                ## Split the dpr sequence output
                sequence_output = dpr_out.hidden_states[-1]
                attn_mask = self.get_attn_mask(input_ids)
                ## Split sequence output, and pool each sequence
                seq_out_0 = []  # last turn, if query; doc structure if passage
                seq_out_1 = []  # dial history, if query; passage text if passage
                dialog_lengths = []
                for i in range(sequence_output.shape[0]):
                    seq_out_masked = sequence_output[i, attn_mask[i], :]
                    segment_masked = token_type_ids[i, attn_mask[i]]
                    seq_out_masked_0 = seq_out_masked[segment_masked == 0, :]
                    seq_out_masked_1 = seq_out_masked[segment_masked == 1, :]
                    dialog_lengths.append((len(seq_out_masked_0), len(seq_out_masked_1)))
                    ### perform pooling
                    seq_out_0.append(self.mean_pool(seq_out_masked_0))
                    seq_out_1.append(self.mean_pool(seq_out_masked_1))

                pooled_output_0 = torch.cat([seq.view(1, -1) for seq in seq_out_0], dim=0)
                pooled_output_1 = torch.cat([seq.view(1, -1) for seq in seq_out_1], dim=0)

                if self.config.scoring_func in ["reranking_original", "current_original"]:
                    current_out = current_turn_output
                else:
                    current_out = pooled_output_0

                out = self.retriever(
                    input_ids,
                    combined_out.cpu().detach().to(torch.float32).numpy(),
                    current_out.cpu().detach().to(torch.float32).numpy(),
                    pooled_output_1.cpu().detach().to(torch.float32).numpy(),
                    prefix=self.generator.config.prefix,
                    n_docs=n_docs,
                    dialog_lengths=dialog_lengths,
                    domain=domain,
                    return_tensors="pt",
                )
            else:
                combined_out = self.question_encoder(input_ids, attention_mask=attention_mask)[0]
                out = self.retriever(
                    input_ids,
                    combined_out.cpu().detach().to(torch.float32).numpy(),
                    combined_out.cpu().detach().to(torch.float32).numpy(),  ## sending dummy
                    combined_out.cpu().detach().to(torch.float32).numpy(),  ## sending dummy
                    prefix=self.generator.config.prefix,
                    n_docs=n_docs,
                    dialog_lengths=dialog_lengths,
                    domain=domain,
                    return_tensors="pt",
                    bm25=self.bm25,
                )

        context_input_ids, context_attention_mask, retrieved_doc_embeds, retrieved_doc_scores = (
                out["context_input_ids"],
                out["context_attention_mask"],
                out["retrieved_doc_embeds"],
                out["doc_scores"],
        )

        # set to correct device
        retrieved_doc_embeds = retrieved_doc_embeds.to(combined_out)
        context_input_ids = context_input_ids.to(input_ids)
        context_attention_mask = context_attention_mask.to(input_ids)
        doc_scores = retrieved_doc_scores.to(combined_out)


        # I don't care about doc scores bc I'm reranking
        