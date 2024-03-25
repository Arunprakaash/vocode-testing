import torch

model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                              model='silero_vad',
                              force_reload=True)


class CustomVADIterator(utils[3]):

    def reset_states(self):
        self.model.reset_states()
        self.triggered = False
        self.temp_end = 0
        self.current_sample = 0
        self.collected_chunks = []

    def __call__(self, x, return_seconds=False):
        """
        x: torch.Tensor
            audio chunk (see examples in repo)

        return_seconds: bool (default - False)
            whether return timestamps in seconds (default - samples)
        """

        if not torch.is_tensor(x):
            try:
                x = torch.Tensor(x)
            except:
                raise TypeError("Audio cannot be casted to tensor. Cast it manually")

        window_size_samples = len(x[0]) if x.dim() == 2 else len(x)
        self.current_sample += window_size_samples

        speech_prob = self.model(x, self.sampling_rate).item()

        if self.triggered:
            self.collected_chunks.append(x)

        if (speech_prob >= self.threshold) and self.temp_end:
            self.temp_end = 0

        if (speech_prob >= self.threshold) and not self.triggered:
            self.triggered = True
            speech_start = self.current_sample - self.speech_pad_samples - window_size_samples
            return {
                'start': int(speech_start) if not return_seconds else round(speech_start / self.sampling_rate, 1)}, None

        if (speech_prob < self.threshold - 0.15) and self.triggered:
            if not self.temp_end:
                self.temp_end = self.current_sample
            if self.current_sample - self.temp_end < self.min_silence_samples:
                return None, None
            else:
                speech_end = self.temp_end + self.speech_pad_samples - window_size_samples
                self.temp_end = 0
                self.triggered = False
                combined_chunk = torch.cat(self.collected_chunks)
                combined_chunk = combined_chunk.detach().cpu().numpy().tobytes()
                self.collected_chunks = []
                return {'end': int(speech_end) if not return_seconds else round(speech_end / self.sampling_rate,
                                                                                1)}, combined_chunk

        return None, None
