import backoff
import requests
import logging
import time

logging.basicConfig(level=logging.INFO)


class ServerError(Exception):
    pass


DEFAULT_LANGUAGE_CODES = {
    "DE": "German",
    "FR": "French",
    "IT": "Italian",
    "EN": "English",
}


class Translator:
    def __init__(
        self,
        api_key,
        model="gpt-4o",
        request_per_second=2,
        language_code=DEFAULT_LANGUAGE_CODES,
    ):
        self.api_key = api_key
        self.requests_per_second = request_per_second
        self.language_code = language_code
        self.model = model

    @backoff.on_exception(
        backoff.expo, (requests.exceptions.RequestException, ServerError), max_time=60
    )
    def translate_segment(self, text, images, source_language, target_language):
        """Translate the audio description for the frames of a video from the source language to the target language.
        args:
            text: str: The audio description to translate
            images: list[str]: A list of base64 encoded frames to send to the model
            source_language: str: The source language code
            target_language: str: The target language code
        return:
            str: The translated audio description
        """

        if (
            source_language not in self.language_code
            or target_language not in self.language_code
        ):
            raise ValueError("Invalid language code")
        source_language = self.language_code[source_language]
        target_language = self.language_code[target_language]

        text = (
            f"Translate the following audio description for the frames of this video from {source_language} to"
            f" {target_language}. Respond with the translation only. If the audio description does not match the "
            f"image, please ignore the image. Respond with a translation only. This is the audio description to translate: \n {text}"
        )

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": [{"type": "text", "text": text}]}],
            "max_tokens": 300,
        }

        logging.info(
            f"Sending request to OpenAI with payload (images not included): {payload}"
        )

        for img in images:
            image = {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img}"},
            }

            payload["messages"][0]["content"].append(image)

        response = requests.post(
            "https://api.openai.com/v1/chat/completions", headers=headers, json=payload
        )

        if response.status_code != 200 or "error" in response.json():
            raise ServerError(f"Error in response: {response.json()}")
        logging.info(f"Received response from OpenAI: {response.json()}")

        time.sleep(1 / self.requests_per_second)

        translation = response.json()["choices"][0]["message"]["content"]
        return translation


if __name__ == "__main__":
    pass
