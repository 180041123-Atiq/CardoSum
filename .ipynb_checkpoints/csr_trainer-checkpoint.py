from trl.trainer.grpo_trainer import *
import numpy as np

class CSRTrainer(GRPOTrainer):
    def __init__(
        self,
        model: "str | PreTrainedModel | PeftModel",
        reward_funcs: RewardFunc | list[RewardFunc],
        args: GRPOConfig | None = None,
        train_dataset: Dataset | IterableDataset | None = None,
        eval_dataset: Dataset | IterableDataset | dict[str, Dataset | IterableDataset] | None = None,
        processing_class: PreTrainedTokenizerBase | ProcessorMixin | None = None,
        reward_processing_classes: PreTrainedTokenizerBase | list[PreTrainedTokenizerBase] | None = None,
        callbacks: list[TrainerCallback] | None = None,
        optimizers: tuple[torch.optim.Optimizer | None, torch.optim.lr_scheduler.LambdaLR | None] = (None, None),
        peft_config: "PeftConfig | None" = None,
        tools: list[Callable] | None = None,
        rollout_func: RolloutFunc | None = None,
        environment_factory: EnvironmentFactory | None = None,
    ):
        super().__init__(
            model=model,
            reward_funcs=reward_funcs,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            reward_processing_classes=reward_processing_classes,
            callbacks=callbacks,
            optimizers=optimizers,
            peft_config=peft_config,
            tools=tools,
            rollout_func=rollout_func,
            environment_factory=environment_factory,
        )

    @profiling_decorator
    def _prepare_inputs(self, generation_batch: dict[str, torch.Tensor | Any]) -> dict[str, torch.Tensor | Any]:
        # Prepares inputs for model training/evaluation by managing completion generation and batch handling.
        # During training:
        #   - Receives the local generation batch (Per-GPU batch size × steps per generation)
        #     from the modified training dataloader instead of the standard local batch
        #   - Generates completions once for the entire generation batch and splits it into batches of size
        #     `per_device_train_batch_size`
        #   - Buffers these completions and returns the appropriate slice for the current accumulation step
        #   - Optimizes by regenerating completions only periodically (every steps_per_generation * num_iterations)
        # During evaluation:
        #   - The input is treated as a standard local batch (no accumulation, no multiple iterations)
        #   - Completions are generated for each batch without buffering or reuse
        # Returns a single local batch in both cases.

        mode = "train" if self.model.training else "eval"
        if mode == "train":
            generate_every = self.args.steps_per_generation * self.num_iterations
            if self._step % generate_every == 0 or self._buffered_inputs is None:
                # self._buffered_inputs=None can occur when resuming from a checkpoint
                generation_batch = self._generate_and_score_completions(generation_batch)
                generation_batch = split_pixel_values_by_grid(generation_batch)
                generation_batch = shuffle_sequence_dict(generation_batch)
                generation_batches = split_tensor_dict(generation_batch, self.args.steps_per_generation)
                self._buffered_inputs = [unsplit_pixel_values_by_grid(batch) for batch in generation_batches]
            inputs = self._buffered_inputs[self._step % self.args.steps_per_generation]
        else:
            # In evaluation, there is neither batch grouping for generation, nor multiple iterations, hence
            # local generation batch == local eval batch
            inputs = self._generate_and_score_completions(generation_batch)

        print(type(inputs))
        print(inputs.keys())
        print(type(inputs['prompt_ids']))
        print(inputs['prompt_ids'].size())
        print(inputs['completion_ids'].size())
        exit()
        
        return inputs
    
    def _generate_and_score_completions(
        self, inputs: list[dict[str, torch.Tensor | Any]]
    ) -> dict[str, torch.Tensor | Any]:
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"

        prompts = [x["prompt"] for x in inputs]

        if self.environments:
            for prompt, environment, reset_kwargs in zip(prompts, self.environments, inputs, strict=True):
                observation = environment.reset(**reset_kwargs)
                if observation is None:
                    continue
                prompt[-1]["content"] += observation

        if "images" in inputs[0]:
            images = [example.get("images") for example in inputs]
        elif "image" in inputs[0]:
            images = [[example.get("image")] if example.get("image") is not None else None for example in inputs]
        else:
            images = None
        # Transformers requires at least one image in the batch, otherwise it throws an error
        if images is not None and all(img_list == [] for img_list in images):
            images = None

        # If the prompts are conversational and the inputs contain images, we need to convert the prompts from
        # [{"role": "user", "content": "What color is the sky?"}] to
        # [{"role": "user", "content": [{"type": "image", "image": <Image>}, {"type": "text", "text": "What color is the sky?"}]}]
        if images is not None:
            if not is_conversational(inputs[0]):
                raise ValueError(
                    "Multimodal training requires conversational prompts. It looks like the dataset contains "
                    "non-conversational inputs, likely because a chat template was applied before passing the dataset "
                    "to the trainer. Please provide the raw conversational prompts and let the trainer apply the chat "
                    "template internally."
                )
            prompts = [
                prepare_multimodal_messages(prompt, image_list)
                for prompt, image_list in zip(prompts, images, strict=True)
            ]

        (
            prompt_ids_list,
            completion_ids_list,
            tool_mask_list,
            completions,
            num_items_in_batch,
            sampling_per_token_logps_list,
            extra_fields,
        ) = self._generate(prompts)

        # print("num_generations:", self.num_generations) # G => 4
        # print("input batch size:", len(inputs)) # B => 4, since batch size is 4
        # print("after generation prompts shape:", np.array(prompt_ids_list).shape) # (B,..) => (4,345)
        # print("after generation completion shape:", np.array(completion_ids_list).shape) # (B*G, ...) => (16,64)
        # exit()
        
        # broad casting the prompt so that it has size (B*G,...)
        # prompt_ids_list = [
        #     p for p in prompt_ids_list for _ in range(self.num_generations)
        # ]

        # Convert lists of token IDs to padded tensors
        prompt_ids = [torch.tensor(ids) for ids in prompt_ids_list]
        prompt_mask = [torch.ones_like(ids, dtype=torch.long) for ids in prompt_ids]
        prompt_ids = pad(
            prompt_ids,
            padding_value=self.pad_token_id,
            padding_side="left",
            pad_to_multiple_of=self.pad_to_multiple_of,
        ).to(device=device)
        prompt_mask = pad(
            prompt_mask, padding_value=0, padding_side="left", pad_to_multiple_of=self.pad_to_multiple_of
        ).to(device=device)
        completion_ids = [torch.tensor(ids) for ids in completion_ids_list]
        completion_mask = [torch.ones_like(ids, dtype=torch.long) for ids in completion_ids]
        completion_ids = pad(
            completion_ids,
            padding_value=self.pad_token_id,
            padding_side="right",
            pad_to_multiple_of=self.pad_to_multiple_of,
        ).to(device=device)
        completion_mask = pad(
            completion_mask, padding_value=0, padding_side="right", pad_to_multiple_of=self.pad_to_multiple_of
        ).to(device=device)
        if sampling_per_token_logps_list is not None:
            sampling_per_token_logps = [torch.tensor(logps) for logps in sampling_per_token_logps_list]
            sampling_per_token_logps = pad(
                sampling_per_token_logps,
                padding_value=0.0,
                padding_side="right",
                pad_to_multiple_of=self.pad_to_multiple_of,
            ).to(device=device)
        else:
            sampling_per_token_logps = None
        if tool_mask_list is not None:
            tool_mask = [torch.tensor(mask) for mask in tool_mask_list]
            tool_mask = pad(
                tool_mask, padding_value=1, padding_side="right", pad_to_multiple_of=self.pad_to_multiple_of
            ).to(device=device)
        else:
            tool_mask = None

        # If mask_truncated_completions is enabled, zero out truncated completions for attention and loss masking
        if self.mask_truncated_completions:
            eos_and_pad = [self.eos_token_id, self.pad_token_id]
            is_truncated = torch.tensor([ids[-1] not in eos_and_pad for ids in completion_ids_list], device=device)
            # Mask completion_mask for attention masking
            completion_mask = completion_mask * (~is_truncated).unsqueeze(1).int()
            # Also mask tool_mask for consistency in multi-turn training
            if tool_mask is not None:
                tool_mask = tool_mask * (~is_truncated).unsqueeze(1).int()

        # Concatenate prompt_mask with completion_mask for logit computation
        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)  # (B, P+C)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B, P+C)

        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens
        batch_size = self.args.per_device_train_batch_size if mode == "train" else self.args.per_device_eval_batch_size

        num_images = [len(img_list) for img_list in images] if images is not None else None

        # Get forward_kwargs for models with multimodal inputs
        if images is not None:
            prompts_text = [
                apply_chat_template(
                    {"prompt": prompt}, self.processing_class, tools=self.tools, **self.chat_template_kwargs
                )["prompt"]
                for prompt in prompts
            ]
            prompt_inputs = self.processing_class(images=images, text=prompts_text, padding=True, return_tensors="pt")
            prompt_inputs = super()._prepare_inputs(prompt_inputs)
            forward_kwargs = {k: v for k, v in prompt_inputs.items() if k not in ["input_ids", "attention_mask"]}
        else:
            forward_kwargs = {}

        # If token_type_ids are used, extend them with zeros for the completion part
        if "token_type_ids" in forward_kwargs:
            token_type_ids = forward_kwargs["token_type_ids"]
            if self.pad_to_multiple_of is not None:
                # Needed only with pad_to_multiple_of: otherwise prompt_ids and token_type_ids must have equal len
                padding_size = prompt_ids.size(1) - token_type_ids.size(1)
                if padding_size > 0:
                    token_type_ids = torch.cat(
                        [token_type_ids.new_zeros((token_type_ids.size(0), padding_size)), token_type_ids], dim=1
                    )
            forward_kwargs["token_type_ids"] = torch.cat(
                [token_type_ids, token_type_ids.new_zeros(completion_ids.shape)], dim=1
            )
        # If mm_token_type_ids are used, extend them with zeros for the completion part
        if "mm_token_type_ids" in forward_kwargs:
            mm_token_type_ids = forward_kwargs["mm_token_type_ids"]
            if self.pad_to_multiple_of is not None:
                # Needed only with pad_to_multiple_of: otherwise prompt_ids and mm_token_type_ids must have equal len
                padding_size = prompt_ids.size(1) - mm_token_type_ids.size(1)
                if padding_size > 0:
                    mm_token_type_ids = torch.cat(
                        [mm_token_type_ids.new_zeros((mm_token_type_ids.size(0), padding_size)), mm_token_type_ids],
                        dim=1,
                    )
            forward_kwargs["mm_token_type_ids"] = torch.cat(
                [mm_token_type_ids, mm_token_type_ids.new_zeros(completion_ids.shape)], dim=1
            )

        # When gradient checkpointing is enabled with use_reentrant=True (non default), calling the model inside a
        # torch.no_grad() block triggers a harmless PyTorch warning ("None of the inputs have requires_grad=True").
        # Temporarily disable checkpointing to avoid this warning during inference.
        with torch.no_grad(), disable_gradient_checkpointing(self.model, self.args.gradient_checkpointing_kwargs):
            # If the generation and optimization steps are misaligned—i.e., if generation does not occur at the end of
            # a full optimizer step (when gradient_accumulation_steps is not a multiple of generate_every)—then the
            # samples may come from an earlier version of the model. In that case, we need to track old_per_token_logps
            # for importance sampling. If the steps are aligned, importance sampling isn't necessary and we set
            # old_per_token_logps to None.
            # When using vLLM, we always compute old_per_token_logps for importance sampling, it was shown that the
            # distribution mismatch between vLLM and the training model can be large and harm the training.
            generate_every = self.args.steps_per_generation * self.num_iterations  # generation frequency
            if self.args.gradient_accumulation_steps % generate_every != 0 or (
                self.use_vllm and self.vllm_importance_sampling_correction
            ):
                old_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                    self.model,
                    prompt_completion_ids,
                    attention_mask,
                    logits_to_keep,
                    batch_size,
                    num_images=num_images,
                    **forward_kwargs,  # may contain pixel_values, image_grid_thw, pixel_attention_mask, image_sizes, pixel_position_ids
                )
            else:
                old_per_token_logps = None

            # print('I am here')
            # exit()
            
            # Compute the importance sampling ratio when using vLLM, to correct for potential distribution mismatch
            if self.use_vllm and self.vllm_importance_sampling_correction:
                mask = completion_mask if tool_mask is None else completion_mask * tool_mask
                per_token_logps_diff = (old_per_token_logps - sampling_per_token_logps) * mask

                sequence_level_is = self.vllm_importance_sampling_mode in ["sequence_mask", "sequence_truncate"]
                if sequence_level_is:
                    per_sequence_logps_diff = per_token_logps_diff.sum(dim=-1, keepdim=True)
                    logps_diff = per_sequence_logps_diff
                else:
                    logps_diff = per_token_logps_diff

                vllm_importance_sampling_ratio = torch.exp(logps_diff)

                # vllm_importance_sampling_ratio.shape:
                #   token_* modes:     (B, T)  (per-token ratio)
                #   sequence_* modes:  (B, 1)  (per-sequence ratio)

                if self.vllm_importance_sampling_mode in ["sequence_truncate", "token_truncate"]:
                    vllm_importance_sampling_ratio = torch.clamp(
                        vllm_importance_sampling_ratio, max=self.vllm_importance_sampling_cap
                    )
                elif self.vllm_importance_sampling_mode in ["sequence_mask", "token_mask"]:
                    vllm_importance_sampling_ratio = vllm_importance_sampling_ratio.masked_fill(
                        vllm_importance_sampling_ratio > self.vllm_importance_sampling_cap, value=0.0
                    )
                else:
                    raise ValueError(
                        f"Unknown vLLM importance sampling level: {self.vllm_importance_sampling_mode}. Possible values are 'token_truncate', 'token_mask', 'sequence_truncate', and 'sequence_mask'."
                    )

            # Compute the per-token log probabilities for the reference model
            if self.beta != 0.0:
                if self.ref_model is not None:
                    ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                        self.ref_model,
                        prompt_completion_ids,
                        attention_mask,
                        logits_to_keep,
                        batch_size=batch_size,
                        num_images=num_images,
                        **forward_kwargs,  # may contain pixel_values, image_grid_thw, pixel_attention_mask, image_sizes, pixel_position_ids
                    )
                else:
                    # When training a PEFT adapter, how we obtain the reference depends on the setup:
                    # - New adapter: disabling adapters yields the base model.
                    # - Re-training an existing adapter: an initial copy is loaded under the name "ref".
                    model = self.accelerator.unwrap_model(self.model)
                    with use_adapter(model, adapter_name="ref" if "ref" in model.peft_config else None):
                        ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                            self.model,
                            prompt_completion_ids,
                            attention_mask,
                            logits_to_keep,
                            batch_size=batch_size,
                            num_images=num_images,
                            **forward_kwargs,  # may contain pixel_values, image_grid_thw, pixel_attention_mask, image_sizes, pixel_position_ids
                        )
            else:
                ref_per_token_logps = None

        
        # print('I am here 2')
        # exit()
        
        # Decode
        prompts_text = self.processing_class.batch_decode(prompt_ids, skip_special_tokens=True)
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)

        # print(type(prompts_text))
        # print(np.array(prompts_text).shape)
        # print(np.array(completions_text).shape)
        # exit()

        # Merge extra_fields from rollout_func into inputs for reward functions
        if extra_fields:
            for i, inp in enumerate(inputs):
                for key, values in extra_fields.items():
                    if isinstance(values, list) and i < len(values):
                        inp[key] = values[i]
                    elif not isinstance(values, list):
                        inp[key] = values

        # Calculate rewards for each reward function. rewards_per_func aggregates rewards across all processes. This is
        # important because rewards will be normalized per group, and completions are distributed. We will later slice
        # rewards_per_func to extract each process's subset.
        rewards_per_func = self._calculate_rewards(inputs, prompts, completions, completion_ids_list)
        num_generations = self.num_generations if mode == "train" else self.num_generations_eval

        # print('I am here 3')
        # exit()

        if self.multi_objective_aggregation == "sum_then_normalize":
            # Apply weights to each reward function's output and sum
            rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)
            mean_grouped_rewards = rewards.view(-1, num_generations).mean(dim=1)
            mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(num_generations, dim=0)
            if self.scale_rewards in ["group", "none"]:
                # If self.scale_rewards = "none", we'll only use std_rewards to check for zero std for logging
                if num_generations > 1:
                    std_rewards = rewards.view(-1, num_generations).std(dim=1)
                    std_rewards = std_rewards.repeat_interleave(num_generations, dim=0)
                else:  # doesn't occur during training, but could occur in eval when num_generations_eval=1
                    std_rewards = torch.zeros_like(rewards)
            elif self.scale_rewards == "batch":
                # Compute global std
                if rewards.numel() > 1:
                    std_rewards = rewards.std().expand_as(rewards)
                else:  # doesn't occur during training, but could occur in eval when num_generations_eval=batch_size=1
                    std_rewards = torch.zeros_like(rewards)
            else:
                raise ValueError(
                    f"Invalid value for scale_rewards: {self.scale_rewards}. Must be one of 'batch', 'group', or 'none'."
                )

            advantages = rewards - mean_grouped_rewards
            if self.scale_rewards != "none":
                advantages = advantages / (std_rewards + 1e-4)
            is_std_zero = torch.isclose(std_rewards, torch.zeros_like(std_rewards))  # for logging

        elif self.multi_objective_aggregation == "normalize_then_sum":
            grouped = rewards_per_func.view(-1, num_generations, len(self.reward_funcs))
            mean_k = torch.nanmean(grouped, dim=1, keepdim=True)
            std_k = nanstd(grouped, dim=1, keepdim=True) if num_generations > 1 else torch.zeros_like(mean_k)
            reward_k = (grouped - mean_k) / (std_k + 1e-4)
            reward_k = reward_k.view(-1, len(self.reward_funcs))
            rewards = (reward_k * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)
            std_rewards = rewards.std().expand_as(rewards) if rewards.numel() > 1 else torch.zeros_like(rewards)
            advantages = (rewards - rewards.mean()) / (std_rewards + 1e-4)
            is_std_zero = torch.isclose(std_rewards, torch.zeros_like(std_rewards))  # for logging

        else:
            raise ValueError(
                f"Invalid multi_objective_aggregation: {self.multi_objective_aggregation}. Must be "
                "'sum_then_normalize' or 'normalize_then_sum'."
            )

        # print('I am here 4')
        # exit()
        
        # Slice to keep only the local part of the data
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        all_process_advantages = advantages.clone()  # keep the aggregated advantages for logging
        advantages = advantages[process_slice]

        # Calculate mean reward per function, but only for samples where the function was applied (non-NaN values)
        for i, reward_func_name in enumerate(self.reward_func_names):
            mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/mean"].append(mean_rewards)
            std_func_rewards = nanstd(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/std"].append(std_func_rewards)
        rewards = (rewards_per_func * self.reward_weights.to(rewards_per_func.device).unsqueeze(0)).nansum(dim=1)
        self._metrics[mode]["reward"].append(rewards.mean().item())
        self._metrics[mode]["reward_std"].append(rewards.std().item())
        self._metrics[mode]["frac_reward_zero_std"].append(is_std_zero.float().mean().item())

        # Log prompt and completion texts
        self._logs["prompt"].extend(gather_object(prompts_text))
        self._logs["completion"].extend(gather_object(completions_text))
        for i, name in enumerate(self.reward_func_names):
            self._logs["rewards"][name].extend(rewards_per_func[:, i].tolist())
        self._logs["advantages"].extend(all_process_advantages.tolist())

        # Flush user-logged extra columns (from log_extra), gathering across processes.
        # Keys must be sorted so that all ranks call gather_object in the same order, otherwise values
        # get mis-attributed across columns (dict insertion order may differ between processes).
        for column in sorted(self._pending_extra_logs):
            self._logs["extra"][column].extend(gather_object(self._pending_extra_logs[column]))
        self._pending_extra_logs.clear()

        # Flush user-logged metrics (from log_metric), averaging across processes.
        # Keys must be sorted so that all ranks call accelerator.gather in the same order, otherwise values
        # get mis-attributed across metrics (dict insertion order may differ between processes).
        for name in sorted(self._pending_metrics):
            values = self._pending_metrics[name]
            local_mean = sum(values) / len(values)
            global_mean = self.accelerator.gather(torch.tensor(local_mean, device=device)).mean().item()
            self._metrics[mode][name].append(global_mean)
        self._pending_metrics.clear()

        if images is not None:
            self._logs["images"].extend(gather_object(images))

        if self.use_vllm and self.vllm_importance_sampling_correction:
            delta = torch.abs(old_per_token_logps - sampling_per_token_logps)
            mask = completion_mask.bool() if tool_mask is None else (completion_mask * tool_mask).bool()
            delta = delta[mask]
            mean_delta = torch.mean(delta) if delta.numel() > 0 else torch.tensor(0.0, device=device)
            max_delta = torch.max(delta) if delta.numel() > 0 else torch.tensor(0.0, device=device)
            self._metrics[mode]["sampling/sampling_logp_difference/mean"].append(
                self.accelerator.gather(mean_delta).mean().item()
            )
            self._metrics[mode]["sampling/sampling_logp_difference/max"].append(
                self.accelerator.gather(max_delta).max().item()
            )
            if sequence_level_is:
                flat_is_ratio = vllm_importance_sampling_ratio.flatten()
            else:
                flat_is_ratio = vllm_importance_sampling_ratio[mask]

            min_importance_sampling_ratio = (
                torch.min(flat_is_ratio) if flat_is_ratio.numel() > 0 else torch.tensor(0.0, device=device)
            )
            mean_importance_sampling_ratio = (
                torch.mean(flat_is_ratio) if flat_is_ratio.numel() > 0 else torch.tensor(0.0, device=device)
            )
            max_importance_sampling_ratio = (
                torch.max(flat_is_ratio) if flat_is_ratio.numel() > 0 else torch.tensor(0.0, device=device)
            )
            self._metrics[mode]["sampling/importance_sampling_ratio/min"].append(
                nanmin(self.accelerator.gather(min_importance_sampling_ratio)).item()
            )
            self._metrics[mode]["sampling/importance_sampling_ratio/mean"].append(
                self.accelerator.gather(mean_importance_sampling_ratio).nanmean().item()
            )
            self._metrics[mode]["sampling/importance_sampling_ratio/max"].append(
                nanmax(self.accelerator.gather(max_importance_sampling_ratio)).item()
            )

        # print('I am here 5')
        # exit()
        
        output = {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "advantages": advantages,
            "num_items_in_batch": num_items_in_batch,
        }
        if old_per_token_logps is not None:
            output["old_per_token_logps"] = old_per_token_logps
        if self.use_vllm and self.vllm_importance_sampling_correction:
            output["importance_sampling_ratio"] = vllm_importance_sampling_ratio
        if sampling_per_token_logps is not None:
            output["sampling_per_token_logps"] = sampling_per_token_logps
        if ref_per_token_logps is not None:
            output["ref_per_token_logps"] = ref_per_token_logps
        if "pixel_values" in forward_kwargs:
            output["pixel_values"] = forward_kwargs["pixel_values"]
        if "image_grid_thw" in forward_kwargs:
            output["image_grid_thw"] = forward_kwargs["image_grid_thw"]
        if "pixel_attention_mask" in forward_kwargs:
            output["pixel_attention_mask"] = forward_kwargs["pixel_attention_mask"]
        if "image_sizes" in forward_kwargs:
            output["image_sizes"] = forward_kwargs["image_sizes"]
        if "token_type_ids" in forward_kwargs:
            output["token_type_ids"] = forward_kwargs["token_type_ids"]
        if "mm_token_type_ids" in forward_kwargs:
            output["mm_token_type_ids"] = forward_kwargs["mm_token_type_ids"]
        if "pixel_position_ids" in forward_kwargs:
            output["pixel_position_ids"] = forward_kwargs["pixel_position_ids"]
        if images is not None:
            output["num_images"] = num_images
        if tool_mask is not None:
            output["tool_mask"] = tool_mask

        # print('I am here 6')
        # exit()
        
        return output