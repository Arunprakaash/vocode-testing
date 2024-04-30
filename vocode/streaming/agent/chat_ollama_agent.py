import logging
from typing import Any, Dict, List, Union
from typing import AsyncGenerator, Optional, Tuple

import openai

from vocode import getenv
from vocode.streaming.action.factory import ActionFactory
from vocode.streaming.agent.base_agent import RespondAgent
from vocode.streaming.agent.utils import (
    format_openai_chat_messages_from_transcript,
    collate_response_async,
    openai_get_tokens,
)
from vocode.streaming.models.actions import FunctionCall
from vocode.streaming.models.agent import ChatOllamaAgentConfig
from vocode.streaming.models.transcript import Transcript


class ChatOllamaAgent(RespondAgent[ChatOllamaAgentConfig]):
    def __init__(
            self,
            agent_config: ChatOllamaAgentConfig,
            action_factory: ActionFactory = ActionFactory(),
            logger: Optional[logging.Logger] = None,
            ollama_api_key: Optional[str] = None,
    ):
        super().__init__(
            agent_config=agent_config, action_factory=action_factory, logger=logger
        )

        openai.api_base = agent_config.base_url
        openai.api_key = ollama_api_key or getenv("OLLAMA_API_KEY")
        if not openai.api_key:
            raise ValueError("OLLAMA_API_KEY must be set in environment or passed in")
        self.first_response = (
            self.create_first_response(agent_config.expected_first_prompt)
            if agent_config.expected_first_prompt
            else None
        )
        self.is_first_response = True

    def get_functions(self):
        assert self.agent_config.actions
        if not self.action_factory:
            return None
        return [
            self.action_factory.create_action(action_config).get_openai_function()
            for action_config in self.agent_config.actions
        ]

    def get_chat_parameters(
            self, messages: Optional[List] = None, use_functions: bool = True
    ):
        assert self.transcript is not None
        messages = messages or format_openai_chat_messages_from_transcript(
            self.transcript, self.agent_config.prompt_preamble
        )

        parameters: Dict[str, Any] = {"messages": messages, "max_tokens": self.agent_config.max_tokens,
                                      "temperature": self.agent_config.temperature,
                                      "model": self.agent_config.model_name}

        if use_functions and self.functions:
            parameters["functions"] = self.functions

        return parameters

    def create_first_response(self, first_prompt):
        messages = [
            (
                [{"role": "system", "content": self.agent_config.prompt_preamble}]
                if self.agent_config.prompt_preamble
                else []
            )
            + [{"role": "user", "content": first_prompt}]
        ]

        parameters = self.get_chat_parameters(messages)
        return openai.ChatCompletion.create(**parameters)

    def attach_transcript(self, transcript: Transcript):
        self.transcript = transcript

    async def respond(
            self,
            human_input,
            conversation_id: str,
            is_interrupt: bool = False,
    ) -> Tuple[str, bool]:
        assert self.transcript is not None
        if is_interrupt and self.agent_config.cut_off_response:
            cut_off_response = self.get_cut_off_response()
            return cut_off_response, False
        self.logger.debug("LLM responding to human input")
        if self.is_first_response and self.first_response:
            self.logger.debug("First response is cached")
            self.is_first_response = False
            text = self.first_response
        else:
            chat_parameters = self.get_chat_parameters()
            chat_completion = await openai.ChatCompletion.acreate(**chat_parameters)
            text = chat_completion.choices[0].message.content
        self.logger.debug(f"LLM response: {text}")
        return text, False

    async def generate_response(
            self,
            human_input: str,
            conversation_id: str,
            is_interrupt: bool = False,
    ) -> AsyncGenerator[Tuple[Union[str, FunctionCall], bool], None]:
        if is_interrupt and self.agent_config.cut_off_response:
            cut_off_response = self.get_cut_off_response()
            yield cut_off_response, False
            return
        assert self.transcript is not None

        chat_parameters = {}
        chat_parameters = self.get_chat_parameters()
        chat_parameters["stream"] = True
        stream = await openai.ChatCompletion.acreate(**chat_parameters)
        async for message in collate_response_async(
                openai_get_tokens(stream), get_functions=True
        ):
            yield message, True
