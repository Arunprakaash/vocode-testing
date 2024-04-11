import logging
import os
from fastapi import FastAPI

from vocode.streaming.models.audio_encoding import AudioEncoding
from vocode.streaming.models.synthesizer import ElevenLabsSynthesizerConfig
from vocode.streaming.models.telephony import TwilioConfig
from vocode.streaming.telephony.config_manager.redis_config_manager import (
    RedisConfigManager,
)
from vocode.streaming.models.agent import ChatGPTAgentConfig
from vocode.streaming.models.message import BaseMessage
from vocode.streaming.telephony.server.base import (
    TwilioInboundCallConfig,
    TelephonyServer,
)
from vocode.streaming.models.transcriber import DeepgramTranscriberConfig, PunctuationEndpointingConfig


from speller_agent import SpellerAgentFactory

# if running from python, this will load the local .env
# docker-compose will load the .env file by itself
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(docs_url=None)
import nltk
nltk.download('punkt')
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

config_manager = RedisConfigManager()

BASE_URL = os.getenv("BASE_URL")

# if not BASE_URL:
#     ngrok_auth = os.environ.get("NGROK_AUTH_TOKEN")
#     if ngrok_auth is not None:
#         ngrok.set_auth_token(ngrok_auth)
#     port = sys.argv[sys.argv.index("--port") + 1] if "--port" in sys.argv else 3000
#
#     # Open a ngrok tunnel to the dev server
#     BASE_URL = ngrok.connect(port).public_url.replace("https://", "")
#     logger.info('ngrok tunnel "{}" -> "http://127.0.0.1:{}"'.format(BASE_URL, port))

if not BASE_URL:
    raise ValueError("BASE_URL must be set in environment if not using pyngrok")
synthesizer_config = ElevenLabsSynthesizerConfig(
    api_key="431f452112cab175b80762e50e525c8f",
    model_id="eleven_turbo_v2",
    voice_id="jSkEqGxlKjJ7CUsGgouj",
    sampling_rate=8000,
    audio_encoding=AudioEncoding.MULAW,
    optimize_streaming_latency=4,
    experimental_streaming=True
)

transcriber_config = DeepgramTranscriberConfig(
                language="en-IN",
                sampling_rate=8000,
                model="nova-2",
                audio_encoding=AudioEncoding.MULAW,
                chunk_size=4000,
                endpointing_config=TimeEndpointingConfig(),

)

telephony_server = TelephonyServer(
    base_url=BASE_URL,
    config_manager=config_manager,
    inbound_call_configs=[
        TwilioInboundCallConfig(
            url="/inbound_call",
            agent_config=ChatGPTAgentConfig(
                initial_message=BaseMessage(text="Hi, Welcome to sayvai"),
                prompt_preamble="""## Identity
You are Jennifer from New York Estate company calling user over the phone. You are a pleasant and extremely friendly receptionist caring deeply for the user. Greeting with Sam at the beginning of the call "Hello, Sam. This is Jennifer from New York Estate company. Am I catching you at a good time?"

## Style Guardrails
Embrace Variety: Use diverse language and rephrasing to enhance clarity without repeating content.
Be Conversational: Use everyday language, making the chat feel like talking to a friend.
Be Proactive: Lead the conversation, often wrapping up with a question or next-step suggestion.
Avoid multiple questions in a single response.
Get clarity: If the user only partially answers a question, or if the answer is unclear, keep asking to get clarity.
Use a colloquial way of referring to the date (like Friday, Jan 14th, or Tuesday, Jan 12th, 2024 at 8am).

## Response Guideline
Adapt and Guess: Try to understand transcripts that may contain transcription errors. Avoid mentioning "transcription error" in the response.
Stay in Character: Keep conversations within your role's scope, guiding them back creatively without repeating.
Ensure Fluid Dialogue: Respond in a role-appropriate, direct manner to maintain a smooth conversation flow, be as human as possible, smile, and act very normal.

##Conversational style
Avoid sounding mechanical or artificial; strive for a natural, day-to-day conversational style that makes the clients feel at ease and well-assisted. As the conversation progresses, use filler words such as huh, hmm, ah.""",
                generate_responses=True,
                send_filler_audio=True

            ),
            synthesizer_config=synthesizer_config,
            transcriber_config=transcriber_config,
            twilio_config=TwilioConfig(
                account_sid=os.environ["TWILIO_ACCOUNT_SID"],
                auth_token=os.environ["TWILIO_AUTH_TOKEN"],
            ),
        )
    ],
    # agent_factory=SpellerAgentFactory(),
    logger=logger,
)

app.include_router(telephony_server.get_router())
