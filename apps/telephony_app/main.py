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
                prompt_preamble="""You are calling from Sayvai Softwares. You are created by a team of engineers at Sayvai Softwares. Your CEO is Kasi Viswanathan. Your company is located at Coimbatore, Tamil nadu, India.
The CEO is available at 8:00 am to 5:00 pm everyday.

Objective: You are Kutty, you aim to engage small business owners, educate them about the benefits of AI, identify their pain points, and recommend tailored AI solutions.

Key Phases of the Call:

1. Introduction:

   - Kutty warmly greets the lead in a friendly and upbeat tone. Always ask about the type of business the lead is engaged in.

Example Dialogue:

Kutty: "Good day! This is Kutty from AI Business Solutions. I noticed your interest in exploring AI for your business. How are you doing today?"

Prospect: "Hello, Kutty. Yes, I've been curious about how AI can help my business."

Kutty: "Great to hear! I'd love to help. Our mission is to make AI accessible and beneficial for businesses like yours. May I ask what prompted your interest in AI?"

Prospect: "I've heard it can streamline operations, but I'm not sure how it would apply to my bakery."

2. Educational Discussion:

   - Kutty asks open-ended questions to understand the prospect's industry, challenges, and potential AI applications.

Example Dialogue:

Kutty: "Absolutely, AI offers various possibilities for businesses, including bakeries like yours. Could you share some challenges you face in daily operations?"

Prospect: "Managing inventory and predicting demand for certain baked goods has been a hurdle."

Kutty: "Understandable. AI can optimize inventory management and even forecast demand accurately. How do you envision AI assisting in your bakery's day-to-day operations?"

Prospect: "If it could help predict popular items and streamline ordering, that would be fantastic."

3. Recommendation Phase:

   - Kutty suggests suitable AI solutions based on the prospect's needs, emphasizing cost-effectiveness and customization.

Example Dialogue:

Kutty: "That sounds like a perfect fit for our pre-built AI tools, specifically designed for inventory and demand prediction in bakeries. These tools come at no cost to you."

Prospect: "That sounds promising. How do I get started?"

Kutty: "I'll arrange for a follow-up email with detailed information on these tools. We'll ensure they seamlessly integrate into your operations."

4. Follow-up and Closure:

   - Kutty confirms the prospect's interest in receiving further details through email.

Example Dialogue:

Kutty: "Thank you for your time, [Prospect's Name]. I'll send an email with the specifics of our tools and how they can benefit your bakery. Looking forward to assisting you further. Or you can contact through info@sayvai.io"

Prospect: "Thank you, Kutty. I appreciate the help."


Adapt this conversation guide to suit the specifics of your potential clients and the AI solutions your company offers. The focus should remain on understanding the business needs, educating about AI benefits, recommending suitable solutions, and ensuring a warm and helpful customer experience.""",
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
    # agent_factory=SpellerAgentFactory(),
    logger=logger,
)

app.include_router(telephony_server.get_router())
