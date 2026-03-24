from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


class LocalLLM:

    def __init__(self):
        model_name = "mistralai/Mistral-7B-Instruct-v0.2"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )

    def analyze_scene(self, dialogue_units):

        combined_text = ""
        for i, unit in enumerate(dialogue_units):
            combined_text += f"{i+1}. {unit['speaker']}: {unit['dialogue']}\n"

        prompt = f"""
    You are an expert film director AI.
    
    Analyze each dialogue and return a JSON list.
    
    Each item must contain:
    - index
    - emotion
    - intent
    - shot_type
    - camera_angle
    - camera_movement
    - duration
    
    Dialogues:
    {combined_text}
    
    Return ONLY JSON list.
    """

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=500,
            do_sample=False
        )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        try:
            json_start = response.find("[")
            json_end = response.rfind("]") + 1
            json_str = response[json_start:json_end]
            return eval(json_str)
        except:
            return []