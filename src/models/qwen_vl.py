from typing import Any, Callable, Dict, List

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

from .image_text_model import ImageTextModel

__all__ = ["QwenVL"]


class QwenVL(ImageTextModel):

    def set_model(
        self,
        cache_dir:str = None,
    ) -> None:

        self.model_ = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_name_or_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            local_files_only=self.local_files_only,
            cache_dir=cache_dir,
        )

    def get_language_model(
        self,
    ) -> Callable:

        return self.model_.model

    def get_lm_head(
        self,
    ) -> Callable:

        return self.model_.lm_head

    def set_processor(
        self,
    ) -> None:

        self.processor_ = AutoProcessor.from_pretrained(
            self.processor_name,
            local_files_only=self.local_files_only,
            cache_dir=self.cache_dir,
        )
        self.tokenizer_ = self.processor_.tokenizer

    def set_preprocessor(
        self,
    ) -> None:

        self.preprocessor_ = self.preprocess_input

    def get_conversation_round(
        self,
        instruction: str = "What are these?",
        response: str = "",
        image_file: str = "",
    ) -> List[Dict[str, Any]]:

        conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_file,
                    },
                    {"type": "text", "text": instruction},
                ],
            },
        ]
        if response:
            conversation.append(
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": response},
                    ],
                },
            )

        return conversation

    def get_conversation_template(
        self,
        instruction: str = "What are these?",
        response: str = "",
        image_file: str = "",
        **kwargs: Any,
    ) -> Dict[str, Any]:

        conversation = self.get_conversation_round(
            instruction=instruction,
            response=response,
            image_file=image_file,
        )
        return conversation


    def preprocess_text(
        self,
        conversation,
        generation_mode: bool = False,
        continue_final_message: bool = False,
        **kwargs: Any,
    ) -> str:
        add_generation_prompt = generation_mode if not continue_final_message else False


        prompt = self.processor_.apply_chat_template(
            conversation,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=continue_final_message,
            tokenize=False,
        )

        return prompt

    def preprocess_images(
        self,
        conversation,
        **kwargs: Any,
    ) -> List:

        image_inputs, video_inputs = process_vision_info(conversation)

        return image_inputs, video_inputs

    def preprocess_input(
        self,
        instruction: str = "What are these?",
        image_file: str = None,
        response: str = "",
        generation_mode: bool = False,
        continue_final_message: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:

        conversation = self.get_conversation_template(
            instruction=instruction,
            response=response,
            image_file=image_file,
        )

        image, video = self.preprocess_images(conversation)
        text = self.preprocess_text(
            conversation,
            generation_mode=generation_mode,
            continue_final_message=continue_final_message,
        )

        inputs = self.processor_(
            text=[text],
            images=image,
            videos=video,
            padding=True,
            return_tensors="pt",
        )

        return inputs



    def get_hidden_size(
        self,
    ) -> Callable:

        return self.model_.config.get_text_config().hidden_size    


