import asyncio
import logging
import signal

from dotenv import load_dotenv

from vocode.helpers import create_streaming_microphone_input_and_speaker_output
from vocode.streaming.agent import *
from vocode.streaming.models.agent import *
from vocode.streaming.models.message import BaseMessage
from vocode.streaming.models.synthesizer import *
from vocode.streaming.models.transcriber import *
from vocode.streaming.streaming_conversation import StreamingConversation
from vocode.streaming.synthesizer import *
from vocode.streaming.transcriber import *

load_dotenv()

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


async def main():
    (
        microphone_input,
        speaker_output,
    ) = create_streaming_microphone_input_and_speaker_output(
        use_default_devices=False,
        logger=logger,
        use_blocking_speaker_output=True,

    )

    conversation = StreamingConversation(
        output_device=speaker_output,
        transcriber=DeepgramTranscriber(
            DeepgramTranscriberConfig.from_input_device(
                microphone_input,
                endpointing_config=TimeEndpointingConfig(),
                # model="enhanced-general",
                # language="ta"
            )
        ),
        agent=ChatOllamaAgent(
            ChatOllamaAgentConfig(
                initial_message=BaseMessage(
                    text="வணக்கம்! இது சய்வை சாப்ட்வேர் எல்எல்பி சலுகைகளின் ஷருக் ஆகும். "
                         "உங்கள் தீவிரவாதம் உங்கள் செயல்பாட்டிற்கான ஐயை பற்றிய கவனம் கிடைத்ததாக காண்கிறேன். "
                         "இன்று நீங்கள் எப்படி இருக்கிறீர்கள்?"
                ),
                prompt_preamble="""respond in Colloquial english.""",
                send_filler_audio=True,
                model_name="conceptsintamil/tamil-llama-7b-instruct-v0.2:latest",
                base_url="http://164.52.196.188:3000/v1"
            )
        ),
        synthesizer=ElevenLabsSynthesizer(
            ElevenLabsSynthesizerConfig.from_output_audio_config(speaker_output,
                                                                 api_key="431f452112cab175b80762e50e525c8f",
                                                                 model_id="eleven_multilingual_v2",
                                                                 optimize_streaming_latency=4,
                                                                 voice_id="OG2paYSf2OmPwGrYVyHb",
                                                                 experimental_streaming=True
                                                                 )),
        logger=logger,
    )
    await conversation.start()
    print("Conversation started, press Ctrl+C to end")
    signal.signal(
        signal.SIGINT, lambda _0, _1: asyncio.create_task(conversation.terminate())
    )
    while conversation.is_active():
        chunk = await microphone_input.get_audio()
        conversation.receive_audio(chunk)


if __name__ == "__main__":
    asyncio.run(main())
