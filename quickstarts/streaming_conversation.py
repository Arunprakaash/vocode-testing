import asyncio
import logging
import signal
from dotenv import load_dotenv

# from vocode.streaming.agent.groq_agent import ChatGroqAgent

load_dotenv()

from vocode.streaming.streaming_conversation import StreamingConversation
from vocode.helpers import create_streaming_microphone_input_and_speaker_output
from vocode.streaming.transcriber import *
from vocode.streaming.agent import *
from vocode.streaming.synthesizer import *
from vocode.streaming.models.transcriber import *
from vocode.streaming.models.agent import *
from vocode.streaming.models.synthesizer import *
from vocode.streaming.models.message import BaseMessage

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
        # this moves the playback to a separate thread, set to False to use the main thread
    )

    conversation = StreamingConversation(
        output_device=speaker_output,
        transcriber=DeepgramTranscriber(
            DeepgramTranscriberConfig.from_input_device(
                microphone_input,
                endpointing_config=TimeEndpointingConfig(),
            )
        ),
        agent=ChatGPTAgent(
            ChatGPTAgentConfig(
                initial_message=BaseMessage(
                    text="Good day! This is Sharukh from Sayvai software LLP solution. "
                         "I noticed your interest in exploring about AI for your solution. "
                         "How are you doing today?"),
                prompt_preamble="""Imagine you're welcoming an old friend into your home – warm, inviting, and genuinely
                 excited to catch up. Now, transfer that same energy into your voice as you greet the customer. Start 
                 with a friendly 'Hello!' or 'Hi there!' and remember to smile, even though they can't see you. As you 
                 introduce yourself, use a tone that's relaxed yet enthusiastic, like you're genuinely looking forward 
                 to chatting with them. Once you've established that friendly vibe, dive into the conversation with 
                 curiosity and empathy. Ask open-ended questions about their business challenges and listen attentively 
                 to their responses. Share anec
                 dotes or success stories from other clients to showcase the real-world 
                 impact of your AI solutions. And most importantly, be authentic – let your passion for helping 
                 customers shine through in every word you say. With this approach, you'll not only build rapport but 
                 also create a memorable and positive experience for the customer.""",
                send_filler_audio=True
            )
        ),
        synthesizer=ElevenLabsSynthesizer(ElevenLabsSynthesizerConfig.from_output_audio_config(speaker_output,
                                                                                               api_key="431f452112cab175b80762e50e525c8f",
                                                                                               model_id="eleven_turbo_v2",
                                                                                               optimize_streaming_latency=4,
                                                                                               voice_id="YhmvnzBrCUHpNOlY18BG",
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
