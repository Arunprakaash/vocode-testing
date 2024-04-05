import logging
from typing import List, Text, Optional

from vocode.streaming.action.factory import ActionFactory
from vocode.streaming.agent.base_agent import RespondAgent
from vocode.streaming.models.agent import AgentConfig, ChatGPTAgentConfig


class LangchainAgentConfig(AgentConfig, type="langchain_agent"):
    tools_needed: List[Text]



class LagchainAgent(RespondAgent[LangchainAgentConfig])
    def __init__(self,
                 agent_config: ChatGPTAgentConfig,
                 action_factory: ActionFactory = ActionFactory(),
                 logger: Optional[logging.Logger] = None,
                 openai_api_key: Optional[str] = None,
                 ):
        super().__init__(
            agent_config=agent_config, action_factory=action_factory, logger=logger
        )

