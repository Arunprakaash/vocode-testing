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
                initial_message=BaseMessage(text="Good day! This is Kutty from AI Business Solutions. I noticed your "
                                                 "interest in exploring AI for your business. How are you doing "
                                                 "today?"),
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


Adapt this conversation guide to suit the specifics of your potential clients and the AI solutions your company 
offers. The focus should remain on understanding the business needs, educating about AI benefits, recommending 
suitable solutions, and ensuring a warm and helpful customer experience.""",
                send_filler_audio=True
            )
        ),
        synthesizer=ElevenLabsSynthesizer(ElevenLabsSynthesizerConfig.from_output_audio_config(speaker_output,
                                                                                               api_key="431f452112cab175b80762e50e525c8f",
                                                                                               model_id="eleven_turbo_v2",
                                                                                               optimize_streaming_latency=4,
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
