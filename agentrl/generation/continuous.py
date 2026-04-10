"""Continuous batching rollout orchestration for AgentRL."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch

from agentrl.core.base import BaseEnvironment
from agentrl.core.rollout import RolloutBatch, RolloutOrchestrator


LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class _EpisodeState:
    """Mutable per-episode state used by the continuous scheduler."""

    env: BaseEnvironment
    prompt_text: str
    observations: list[str]
    actions: list[str]
    done: bool = False
    truncated: bool = False
    reward: float = 0.0
    transcript_text: str = ""
    assistant_spans: list[tuple[int, int]] | None = None


class ContinuousBatchingOrchestrator(RolloutOrchestrator):
    """Rollout orchestrator that drops finished sequences during decoding."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._high_padding_streak = 0

    def collect(self) -> RolloutBatch:
        """Collect one rollout batch using continuous per-step scheduling."""

        states: list[_EpisodeState] = []
        for _ in range(self.config.batch_size):
            root_env = self._clone_environment(self.environment)
            initial_observation = root_env.reset()
            for _ in range(self.config.group_size):
                states.append(
                    _EpisodeState(
                        env=self._clone_environment(root_env),
                        prompt_text=initial_observation,
                        observations=[initial_observation],
                        actions=[],
                    )
                )

        padding_ratios: list[float] = []
        for turn_index in range(self.config.max_episode_steps):
            active_indices = [index for index, state in enumerate(states) if not state.done]
            if not active_indices:
                break

            prompts = [
                self._render_generation_prompt(states[index].observations, states[index].actions)
                for index in active_indices
            ]
            responses, padding_ratio = self._generate_active_batch(prompts)
            padding_ratios.append(padding_ratio)
            self._track_padding_ratio(padding_ratio)

            for state_index, response_text in zip(active_indices, responses, strict=True):
                state = states[state_index]
                state.actions.append(response_text)
                next_observation, done = state.env.step(response_text)
                if done:
                    state.done = True
                    state.reward = float(self.verifier.verify(response_text, state.env.state()))
                    transcript_text, spans = self._render_transcript(state.observations, state.actions)
                    state.transcript_text = transcript_text
                    state.assistant_spans = spans
                else:
                    state.observations.append(next_observation)

        for state in states:
            if state.done:
                continue
            state.truncated = True
            state.reward = float(self.verifier.verify(state.actions[-1] if state.actions else "", state.env.state()))
            state.transcript_text, state.assistant_spans = self._render_transcript(state.observations, state.actions)
            LOGGER.warning(
                "Episode hit max_episode_steps=%s before environment termination.",
                self.config.max_episode_steps,
            )

        episode_dicts = [
            {
                "prompt_text": state.prompt_text,
                "final_response": state.actions[-1] if state.actions else "",
                "responses": list(state.actions),
                "observations": list(state.observations),
                "reward": state.reward,
                "done": state.done,
                "truncated": state.truncated,
                "transcript_text": state.transcript_text,
                "assistant_spans": state.assistant_spans or [],
            }
            for state in states
        ]

        input_ids, attention_mask, action_mask = self._pack_sequences(episode_dicts)
        flat_input_ids = input_ids.view(-1, input_ids.shape[-1])
        flat_attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
        flat_action_mask = action_mask.view(-1, action_mask.shape[-1])

        model_config = getattr(self.layout.model, "config", None)
        if model_config is not None:
            model_config.use_cache = False

        with torch.no_grad():
            policy_sequences = self._compute_logprobs(
                self.layout.policy_forward,
                flat_input_ids,
                flat_attention_mask,
                flat_action_mask,
            )
            ref_sequences = self._compute_logprobs(
                self.layout.reference_forward,
                flat_input_ids,
                flat_attention_mask,
                flat_action_mask,
            )

        rewards = torch.tensor(
            [[episode["reward"] for episode in episode_dicts[i : i + self.config.group_size]]
             for i in range(0, len(episode_dicts), self.config.group_size)],
            dtype=torch.float32,
            device=self.device,
        )
        advantages = self._compute_advantages(rewards)
        metadata = self._build_metadata(episode_dicts, rewards)
        metadata["padding_ratio"] = float(sum(padding_ratios) / len(padding_ratios)) if padding_ratios else 0.0

        return RolloutBatch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            action_mask=action_mask,
            policy_logprobs=policy_sequences.view_as(input_ids),
            ref_logprobs=ref_sequences.view_as(input_ids),
            rewards=rewards,
            advantages=advantages,
            metadata=metadata,
        )

    def _generate_active_batch(self, prompt_texts: list[str]) -> tuple[list[str], float]:
        """Generate responses for active prompts with step-level dynamic batching."""

        encoded_inputs = [
            self.tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
            for prompt_text in prompt_texts
        ]
        prompt_ids = [encoded["input_ids"][0].to(self.device) for encoded in encoded_inputs]
        current_ids = [prompt.clone() for prompt in prompt_ids]
        generated_ids = [torch.empty(0, dtype=torch.long, device=self.device) for _ in prompt_texts]
        finished = [False for _ in prompt_texts]

        total_padding_tokens = 0
        total_step_tokens = 0

        for _ in range(self.config.max_new_tokens):
            active_indices = [index for index, is_finished in enumerate(finished) if not is_finished]
            if not active_indices:
                break

            active_sequences = [current_ids[index] for index in active_indices]
            max_length = max(sequence.numel() for sequence in active_sequences)
            total_step_tokens += max_length * len(active_sequences)
            total_padding_tokens += sum(max_length - int(sequence.numel()) for sequence in active_sequences)

            next_tokens = self._sample_next_tokens(active_sequences, active_indices)
            for batch_offset, episode_index in enumerate(active_indices):
                token = int(next_tokens[batch_offset].item())
                token_tensor = torch.tensor([token], dtype=torch.long, device=self.device)
                generated_ids[episode_index] = torch.cat((generated_ids[episode_index], token_tensor), dim=0)
                current_ids[episode_index] = torch.cat((current_ids[episode_index], token_tensor), dim=0)
                if token == getattr(self.tokenizer, "eos_token_id", None):
                    finished[episode_index] = True

        decoded = [
            self.tokenizer.decode(tokens, skip_special_tokens=True)
            for tokens in generated_ids
        ]
        padding_ratio = float(total_padding_tokens / total_step_tokens) if total_step_tokens else 0.0
        return decoded, padding_ratio

    def _sample_next_tokens(
        self,
        active_sequences: list[torch.Tensor],
        active_indices: list[int],
    ) -> torch.Tensor:
        """Sample one token for each currently active sequence."""

        generation_model = self.layout.model
        if hasattr(generation_model, "generate_step"):
            sampled = generation_model.generate_step(active_sequences=active_sequences, active_indices=active_indices)
            if isinstance(sampled, torch.Tensor):
                return sampled.to(self.device)
            return torch.tensor(sampled, dtype=torch.long, device=self.device)

        chunk_size = self.config.chunk_size or len(active_sequences)
        chunk_outputs: list[torch.Tensor] = []
        for start in range(0, len(active_sequences), chunk_size):
            chunk_sequences = active_sequences[start : start + chunk_size]
            padded = torch.nn.utils.rnn.pad_sequence(
                chunk_sequences,
                batch_first=True,
                padding_value=int(self.tokenizer.pad_token_id),
            )
            attention_mask = (padded != int(self.tokenizer.pad_token_id)).long()
            logits = generation_model(input_ids=padded, attention_mask=attention_mask).logits
            next_token_logits = logits[:, -1, :]

            if self.config.do_sample and self.config.temperature > 0:
                probs = torch.softmax(next_token_logits / self.config.temperature, dim=-1)
                sampled = torch.multinomial(probs, num_samples=1, generator=self.rng).squeeze(-1)
            else:
                sampled = torch.argmax(next_token_logits, dim=-1)
            chunk_outputs.append(sampled)
        return torch.cat(chunk_outputs, dim=0)

    def _track_padding_ratio(self, padding_ratio: float) -> None:
        """Track sustained padding inefficiency and emit guidance."""

        if padding_ratio > 0.4:
            self._high_padding_streak += 1
        else:
            self._high_padding_streak = 0

        LOGGER.info("padding_ratio=%.4f", padding_ratio)
        if self._high_padding_streak >= 10:
            LOGGER.warning(
                "padding_ratio > 0.4 for 10+ consecutive generation steps. "
                "Consider sorting prompts by estimated length in the next epoch."
            )
