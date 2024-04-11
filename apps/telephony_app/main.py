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

telephony_server = TelephonyServer(
    base_url=BASE_URL,
    config_manager=config_manager,
    inbound_call_configs=[
        TwilioInboundCallConfig(
            url="/inbound_call",
            agent_config=ChatGPTAgentConfig(
                initial_message=BaseMessage(text="Hi, Welcome to sayvai"),
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
                generate_responses=True,
                send_filler_audio=True

            ),
            synthesizer_config=synthesizer_config,
            twilio_config=TwilioConfig(
                account_sid=os.environ["TWILIO_ACCOUNT_SID"],
                auth_token=os.environ["TWILIO_AUTH_TOKEN"],
            ),
        )
    ],
    agent_factory=SpellerAgentFactory(),
    logger=logger,
)

app.include_router(telephony_server.get_router())
