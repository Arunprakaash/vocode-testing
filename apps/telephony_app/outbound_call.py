import os
from dotenv import load_dotenv

from vocode.streaming.models.audio_encoding import AudioEncoding
from vocode.streaming.models.synthesizer import ElevenLabsSynthesizerConfig

load_dotenv()

from vocode.streaming.telephony.conversation.outbound_call import OutboundCall
from vocode.streaming.telephony.config_manager.redis_config_manager import (
    RedisConfigManager,
)

from speller_agent import SpellerAgentConfig

BASE_URL = os.environ["BASE_URL"]


async def main():
    config_manager = RedisConfigManager()

    outbound_call = OutboundCall(
        base_url=BASE_URL,
        to_phone="+918870539376",
        from_phone="+15513054795",
        config_manager=config_manager,
        synthesizer_config=ElevenLabsSynthesizerConfig(
            api_key="431f452112cab175b80762e50e525c8f",
            model_id="eleven_turbo_v2",
            sampling_rate=8000,
            audio_encoding=AudioEncoding.MULAW,
            optimize_streaming_latency=4,
            experimental_streaming=True
        ),
        agent_config=SpellerAgentConfig(generate_responses=False),
    )
    print("Starting outbound call to {}".format(outbound_call.to_phone))
    await outbound_call.start()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
