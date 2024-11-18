def reinforce_loss(self, batch: torch.utils.data.Dataset) -> torch.Tensor:
        """Use reinforce with per-step to compute the loss."""
        sample_data = self.sample_and_generate_details(batch)
        gold_answers = [[answ] * FLAGS.rl_sample_size for answ in batch["gold_answers"]]
        per_step_rewards = self.reward_calculator.compute_per_step_rewards(
            gold_answers,
            sample_data["partial_samples"],
            output_template=self.policy_lm.output_template,
            templated_rewards=True,
            terminal_reward_only=terminal_reward_only,
        )
        per_step_rewards = torch.tensor(per_step_rewards, dtype=torch.float64, device=self.policy_lm.device)

        # if FLAGS.include_policy_ref_kl:
        #     # Compute log likelihood of the reference model for the samples.
        #     cleaned_samples = []
        #     for per_example_samples in sample_data["partial_samples"]:
        #         per_example_cleaned_samples = []
        #         for sample_sequence in per_example_samples:
        #             # Last complete one!
        #             sample_output = sample_sequence[-1]
        #             per_example_cleaned_samples.append(sample_output)
        #         cleaned_samples.append(per_example_cleaned_samples)

        #     num_samples_per_example = FLAGS.rl_sample_size
        #     input_texts = batch["texts"]
        #     row_ids = batch["row_ids"]
        #     expanded_input_texts = list(chain.from_iterable([[text] * num_samples_per_example for text in input_texts]))
        #     expanded_row_ids = list(chain.from_iterable([[row_id] * num_samples_per_example for row_id in row_ids]))
        #     flattened_sample_outputs = list(chain.from_iterable(cleaned_samples))
        #     data = self.ref_policy_lm.prepare_text_for_train(
        #         texts=expanded_input_texts, output_texts=flattened_sample_outputs, row_ids=expanded_row_ids
        #     )
        #     dataset = DictDataset(data)
        #     dataloader = DataLoader(
        #         dataset,
        #         shuffle=False,
        #         batch_size=len(dataset),
        #         num_workers=1,
        #     )
        #     for ref_batch in dataloader:
        #         _, ref_token_log_probs, _, _ = self.ref_policy_lm.train(ref_batch, per_step_scores=True, to_train=False)

        #     # the sequence length in the token_log_probs has been shifted to the right by 1 position.
        #     ref_token_log_probs = ref_token_log_probs.view(batch_size, FLAGS.rl_sample_size, -1)
        #     ref_token_log_probs_arr = []
        #     for b_idx in range(batch_size):
        #         ref_token_log_probs_arr_per_example = []
        #         for s_idx in range(FLAGS.rl_sample_size):
        #             ref_token_log_probs_arr_per_example_per_sample = []
        #             ref_token_log_prob_sequence = ref_token_log_probs[b_idx, s_idx, :]
        #             for ref_token_log_prob in ref_token_log_prob_sequence:
        #                 if ref_token_log_prob.item() != 0.0:
        #                     ref_token_log_probs_arr_per_example_per_sample.append(ref_token_log_prob.item())
        #             ref_token_log_probs_arr_per_example.append(ref_token_log_probs_arr_per_example_per_sample)
        #         ref_token_log_probs_arr.append(ref_token_log_probs_arr_per_example)

        #     # normalize the ref log probs.
        #     normalized_ref_token_log_probs_arr = self.normalize_signals(signals=ref_token_log_probs_arr)
        #     # add ref_log_prob as a new reward.
        #     for b_idx in range(batch_size):
        #         for s_idx in range(FLAGS.rl_sample_size):
        #             sequence_rewards = normalized_per_step_rewards[b_idx][s_idx]
        #             ref_kl_sequence_rewards = normalized_ref_token_log_probs_arr[b_idx][s_idx]
        #             try:
        #                 for seq_idx in range(len(sequence_rewards)):
        #                     normalized_per_step_rewards[b_idx][s_idx][seq_idx] += (
        #                         FLAGS.policy_ref_kl_coef * ref_kl_sequence_rewards[seq_idx]
        #                     )
        #             except Exception:
        #                 print("saeed")
        #                 print(len(sequence_rewards))
        #                 print(len(ref_kl_sequence_rewards))
        #                 print(len(ref_token_log_probs_arr[b_idx][s_idx]))
        #                 print(ref_token_log_probs.size())
        #                 print(sample_data["actual_lens"])
        #                 exit()

        normalized_rewards = normalize_signals(per_step_rewards, normalization_type=FLAGS.reward_normalization_type)
        masks_per_step = torch.where(sample_data["labels_to_consider"] == -100, 0, 1)
        returns = form_returns(normalized_rewards * masks_per_step)
        normalized_returns = normalize_signals(returns, normalization_type=FLAGS.reward_normalization_type)
        normalized_returns = normalized_returns * masks_per_step
        if not FLAGS.with_baseline:
            loss = -torch.mean(torch.mean(torch.sum(sample_data["token_log_ps"] * normalized_returns, dim=2), dim=1), dim=0)
        else:
            max_len = returns.size()[2]
            current_minibatch_baseline_returns = torch.sum(torch.sum(normalized_returns, dim=0), dim=0)
            non_zero_counts = torch.sum(torch.sum(masks_per_step, dim=0), dim=0)
            current_minibatch_baseline_returns = current_minibatch_baseline_returns / non_zero_counts
            old_average = self.baseline_returns[:max_len]
            returns_after_baselines = normalized_returns - old_average
            loss = -torch.mean(
                torch.mean(torch.sum(sample_data["token_log_ps"] * returns_after_baselines * masks_per_step, dim=2), dim=1),
                dim=0,
            )

            # Update the baseline
            alpha = FLAGS.baseline_momentum
            self.baseline_returns[:max_len] = alpha * old_average + (1.0 - alpha) * current_minibatch_baseline_returns

            msg = f"\nbaseline reward used: {self.baseline_returns.tolist()}"
            logging.info(msg)

        # Compute the per-step entropy if requested.
        if FLAGS.compute_per_step_entropy:
            entropy_loss = compute_entropy_loss(
                sample_data["labels_to_consider"],
                sample_data["actual_lens"],
                sample_data["token_log_ps"],
                sample_data["logits"],
            )
            msg = f"\nentropy loss computed: {entropy_loss}"
            logging.info(msg)
            coefficient = FLAGS.entropy_coef
            if FLAGS.include_policy_ref_kl:
                # necessary to train with kl penalty between ref and policy.
                coefficient += FLAGS.policy_ref_kl_coef
            loss += coefficient * entropy_loss

        return loss

    def sample_and_generate_details(
        self, batch: torch.utils.data.Dataset, teacher_forcing_labels: Optional[torch.Tensor] = None, to_train: bool = True
    ) -> Any:
        """Compute per-step information while sampling.

        We can also give prior labels which will compute the logits of
        it. Useful for computing the log-p of reference policy for the
        samples.
        """
        assert FLAGS.rl_sample_size > 1
        num_iterative_calls = FLAGS.rl_sample_size // FLAGS.iterative_chunk_size
        partial_generations = []
        final_log_ps = []
        actual_lens = []
        labels_to_consider = []
        token_final_log_ps = []
        logits = []
        for call_idx in range(num_iterative_calls):
            llm_generation_outputs: List[LLMGenerationOutput] = self.policy_lm.generation_pass(
                batch,
                top_p=FLAGS.train_top_p,
                temperature=FLAGS.train_temperature,
                num_return_sequences=FLAGS.iterative_chunk_size,
                to_train=to_train,
                use_cache=True,
                per_step_scores=True,
                iterative_rl_sampling=False,
                generate_partial_sequences=True,
                teacher_forcing_labels=teacher_forcing_labels,
            )
            final_log_ps_per_call = llm_generation_outputs[0].final_log_ps
            actual_lens_per_call = llm_generation_outputs[0].actual_lens
            labels_to_consider_per_call = llm_generation_outputs[0].labels_to_consider
            token_final_log_ps_per_call = llm_generation_outputs[0].token_final_log_ps
            partial_generations_per_call = llm_generation_outputs[0].partially_generated_sequences
            logits_per_call = llm_generation_outputs[0].logits
            batch_size_extended, seq_len, vocab_size = logits_per_call.size()
            batch_partial_generations_per_call = []
            batch_size = batch_size_extended // FLAGS.iterative_chunk_size
            for b_idx in range(batch_size):
                batch_partial_generations_per_call.append(
                    [
                        partial_generations_per_call[
                            b_idx * FLAGS.iterative_chunk_size : (b_idx + 1) * FLAGS.iterative_chunk_size
                        ]
                    ]
                )
            final_log_ps.append(final_log_ps_per_call.view(batch_size, FLAGS.iterative_chunk_size))
            actual_lens.append(actual_lens_per_call.view(batch_size, FLAGS.iterative_chunk_size))
            labels_to_consider.append(labels_to_consider_per_call.view(batch_size, FLAGS.iterative_chunk_size, -1))
            token_final_log_ps.append(token_final_log_ps_per_call.view(batch_size, FLAGS.iterative_chunk_size, -1))
            logits.append(logits_per_call.view(batch_size, FLAGS.iterative_chunk_size, seq_len, vocab_size))
            partial_generations.append(batch_partial_generations_per_call)
        sequence_log_probs = torch.cat(final_log_ps, dim=1)
        actual_lens_tensor = torch.cat(actual_lens, dim=1)
        batch_size = sequence_log_probs.size()[0]
        final_labels_to_consider = []
        token_log_ps = []
        final_logits = []
        partial_samples = []
        for b_idx in range(batch_size):
            labels_to_consider_arr = []
            token_log_ps_arr = []
            logits_arr = []
            partial_samples_arr = []
            for call_idx in range(num_iterative_calls):
                for chunk_idx in range(FLAGS.iterative_chunk_size):
                    labels_to_consider_arr.append(labels_to_consider[call_idx][b_idx, chunk_idx, :])
                    token_log_ps_arr.append(token_final_log_ps[call_idx][b_idx, chunk_idx, :])
                    logits_arr.append(logits[call_idx][b_idx, chunk_idx, :, :])
                    partial_samples_arr.append(partial_generations[call_idx][b_idx][0][chunk_idx])

            final_labels_to_consider.append(labels_to_consider_arr)
            token_log_ps.append(token_log_ps_arr)
            final_logits.append(logits_arr)
            partial_samples.append(partial_samples_arr)

        max_len = actual_lens_tensor.max()
        # Pad sequences up until max_len
        logits_flattened = []
        token_log_ps_flattened = []
        flattened_labels_to_consider = []
        for b_idx in range(batch_size):
            for sample_idx in range(FLAGS.rl_sample_size):
                # pad labels to consider
                cur_labels_tensor = final_labels_to_consider[b_idx][sample_idx]
                cur_len = cur_labels_tensor.size()[0]
                pad_label_tensor = torch.tensor(
                    [-100] * (max_len - cur_len), dtype=cur_labels_tensor.dtype, device=cur_labels_tensor.device
                )
                flattened_labels_to_consider.append(torch.cat((cur_labels_tensor, pad_label_tensor), dim=0))

                # Pad samples with the last one, corresponding to padding the generated ids with the pad token.
                last_partial_sample = partial_samples[b_idx][sample_idx][-1]
                partial_samples[b_idx][sample_idx].extend([last_partial_sample] * (max_len - cur_len))

                # pad token log_ps
                cur_token_log_ps_tensor = token_log_ps[b_idx][sample_idx]
                pad_token_log_ps_tensor = torch.tensor(
                    [0.0] * (max_len - cur_len), dtype=cur_token_log_ps_tensor.dtype, device=cur_token_log_ps_tensor.device
                )
                token_log_ps_flattened.append(torch.cat((cur_token_log_ps_tensor, pad_token_log_ps_tensor), dim=0))

                # pad logits
                cur_logits = final_logits[b_idx][sample_idx]
                cur_len, vocab_size = cur_logits.size()
                pad_logits = torch.zeros((max_len - cur_len, vocab_size), dtype=cur_logits.dtype, device=cur_logits.device)
                logits_flattened.append(torch.cat((cur_logits, pad_logits), dim=0))

        final_logits = torch.stack(logits_flattened, dim=0).view(batch_size, FLAGS.rl_sample_size, -1, vocab_size)
        final_token_log_ps = torch.stack(token_log_ps_flattened, dim=0).view(batch_size, FLAGS.rl_sample_size, -1)
        final_labels_to_consider = torch.stack(flattened_labels_to_consider, dim=0).view(batch_size, FLAGS.rl_sample_size, -1)

        return_data = {
            "sequence_log_probs": sequence_log_probs,
            "actual_lens": actual_lens_tensor,
            "partial_samples": partial_samples,
            "labels_to_consider": final_labels_to_consider,
            "token_log_ps": final_token_log_ps,
            "logits": final_logits,
        }

        # We should add a padding code to here!
        return return_data