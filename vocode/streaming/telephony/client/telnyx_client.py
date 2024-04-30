from typing import Optional
from telnyx import Telnyx
from telnyx.error import TelnyxError

from vocode.streaming.models.telephony import BaseCallConfig, TelnyxConfig
from vocode.streaming.telephony.client.base_telephony_client import BaseTelephonyClient
from vocode.streaming.telephony.templater import Templater


class TelnyxClient(BaseTelephonyClient):
    def __init__(self, base_url: str, telnyx_config: TelnyxConfig):
        super().__init__(base_url)
        self.telnyx_config = telnyx_config
        self.telnyx_client = Telnyx(telnyx_config.api_key)
        try:
            # Test credentials
            self.telnyx_client.public_key.list()
        except TelnyxError as e:
            raise RuntimeError(
                "Could not create Telnyx client. Invalid credentials"
            ) from e
        self.templater = Templater()

    def get_telephony_config(self):
        return self.telnyx_config

    async def create_call(
        self,
        conversation_id: str,
        to_phone: str,
        from_phone: str,
        record: bool = False,
        digits: Optional[str] = None,
    ) -> str:
        twiml = self.get_connection_twiml(conversation_id=conversation_id)
        call = self.telnyx_client.calls.create(
            connection_id=twiml.id,
            to=to_phone,
            from_=from_phone,
            record=record,
            **self.get_telephony_config().extra_params,
        )
        return call.id

    def get_connection_twiml(self, conversation_id: str):
        return self.templater.get_connection_twiml(
            base_url=self.base_url, call_id=conversation_id
        )

    async def end_call(self, telnyx_call_id):
        response = self.telnyx_client.calls.get(telnyx_call_id).update(
            state="hangup"
        )
        return response.data.get("state") == "hangup"

    def validate_outbound_call(
        self,
        to_phone: str,
        from_phone: str,
        mobile_only: bool = True,
    ):
        if len(to_phone) < 8:
            raise ValueError("Invalid 'to' phone")

        if not mobile_only:
            return
        try:
            number_info = self.telnyx_client.number_lookup.retrieve(to=to_phone).data
            if number_info["portability"]["carrier"]["type"] != "mobile":
                raise ValueError("Can only call mobile phones")
        except TelnyxError as e:
            raise ValueError(f"Error looking up phone number: {e}") from e
