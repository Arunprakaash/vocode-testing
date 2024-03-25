import logging
from typing import AsyncGenerator
from typing import Optional, Tuple

from langchain import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    HumanMessagePromptTemplate,
)
from langchain.schema import ChatMessage, AIMessage, HumanMessage
from langchain_groq import ChatGroq

from vocode import getenv
from vocode.streaming.agent.base_agent import RespondAgent
from vocode.streaming.agent.utils import get_sentence_from_buffer
from vocode.streaming.models.agent import ChatGroqAgentConfig

SENTENCE_ENDINGS = [".", "!", "?"]


class ChatGroqAgent(RespondAgent[ChatGroqAgentConfig]):
    def __init__(
            self,
            agent_config: ChatGroqAgentConfig,
            logger: Optional[logging.Logger] = None,
            groq_api_key: Optional[str] = None,
    ):
        super().__init__(agent_config=agent_config, logger=logger)
        from groq import AsyncGroq

        groq_api_key = groq_api_key or getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError(
                "GROQ_API_KEY must be set in environment or passed in"
            )
        self.prompt = ChatPromptTemplate.from_messages(
            [
                MessagesPlaceholder(variable_name="history"),
                HumanMessagePromptTemplate.from_template("{input}"),
            ]
        )

        self.llm = ChatGroq(
            model_name=agent_config.model_name,
            groq_api_key=groq_api_key,
        )

        self.groq_client = (
            AsyncGroq(api_key=groq_api_key) if agent_config.generate_responses else None
        )

        self.memory = ConversationBufferMemory(return_messages=True)
        self.memory.chat_memory.messages.append(
            HumanMessage(content=self.agent_config.prompt_preamble)
        )
        if agent_config.initial_message:
            self.memory.chat_memory.messages.append(
                AIMessage(content=agent_config.initial_message.text)
            )

        self.conversation = ConversationChain(
            memory=self.memory, prompt=self.prompt, llm=self.llm
        )

    async def respond(
            self,
            human_input,
            conversation_id: str,
            is_interrupt: bool = False,
    ) -> Tuple[str, bool]:
        text = await self.conversation.apredict(input=human_input)
        self.logger.debug(f"LLM response: {text}")
        return text, False

    async def generate_response(
            self,
            human_input,
            conversation_id: str,
            is_interrupt: bool = False,
    ) -> AsyncGenerator[Tuple[str, bool], None]:
        self.memory.chat_memory.messages.append(HumanMessage(content=human_input))

        bot_memory_message = AIMessage(content="")
        self.memory.chat_memory.messages.append(bot_memory_message)
        prompt = self.llm._create_message_dicts(self.memory.chat_memory.messages, None)[0]

        if self.groq_client:
            streamed_response = await self.groq_client.chat.completions.create(
                messages=prompt,
                model=self.agent_config.model_name,
                stream=True,
                max_tokens=self.agent_config.max_tokens_to_sample,
                stop=None
            )

            buffer = ""
            async for completion in streamed_response:
                buffer += completion.choices[0].delta.content
                sentence, remainder = get_sentence_from_buffer(buffer)
                if sentence:
                    bot_memory_message.content = bot_memory_message.content + sentence
                    buffer = remainder
                    yield sentence, True
                continue

    def update_last_bot_message_on_cut_off(self, message: str):
        for memory_message in self.memory.chat_memory.messages[::-1]:
            if (
                    isinstance(memory_message, ChatMessage)
                    and memory_message.role == "assistant"
            ) or isinstance(memory_message, AIMessage):
                memory_message.content = message
                return
