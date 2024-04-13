import logging
from typing import Optional

import aiohttp

from vocode import getenv
from vocode.streaming.agent.bot_sentiment_analyser import BotSentiment
from vocode.streaming.models.message import BaseMessage
from vocode.streaming.models.synthesizer import (
    DeepgramSynthesizerConfig, SynthesizerType,
)
from vocode.streaming.synthesizer.base_synthesizer import (
    BaseSynthesizer,
    SynthesisResult,
    tracer
)

LUNA_VOICE_ID = "aura-luna-en"
DEEPGRAM_BASE_URL = "https://api.deepgram.com/v1/speak"


class DeepgramSynthesizer(BaseSynthesizer[DeepgramSynthesizerConfig]):
    def __init__(
            self,
            synthesizer_config: DeepgramSynthesizerConfig,
            logger: Optional[logging.Logger] = None,
            aiohttp_session: Optional[aiohttp.ClientSession] = None,
    ):
        super().__init__(synthesizer_config, aiohttp_session)

        self.api_key = synthesizer_config.api_key or getenv("DEEPGRAM_API_KEY")
        self.voice = synthesizer_config.voice or LUNA_VOICE_ID
        self.encoding = synthesizer_config.encoding
        self.bitrate = synthesizer_config.bitrate
        self.container = synthesizer_config.container
        self.sampling_rate = synthesizer_config.sampling_rate

    async def create_speech(
            self,
            message: BaseMessage,
            chunk_size: int,
            bot_sentiment: Optional[BotSentiment] = None,
    ) -> SynthesisResult:
        url = DEEPGRAM_BASE_URL + f"?model={self.voice}"

        headers = {
            "Authorization": f"Token {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "text": message.text,
        }

        create_speech_span = tracer.start_span(
            f"synthesizer.{SynthesizerType.DEEPGRAM.value.split('_', 1)[-1]}.create_total"
        )

        session = self.aiohttp_session

        response = await session.request(
            "POST",
            url,
            json=payload,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=15),
        )
