from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


class LocalLLM:

    def __init__(self):
        model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16
        )

    def analyze_scene(self, dialogue_units):

        combined_text = ""
        for i, unit in enumerate(dialogue_units[:10]):  # LIMIT for speed
            combined_text += f"{i+1}. {unit['speaker']}: {unit['dialogue']}\n"

        prompt = f"""
You are an expert film director and cinematic AI system.

Your task is to convert dialogue into a structured Beat Script for a cinematic engine.

--------------------------------------------------

STRICT RULES:

- You MUST follow the Beat Script schema exactly
- You MUST ONLY use values from the provided Cinematic Vocabulary
- Output MUST be valid JSON (no explanation, no text outside JSON)
- Each dialogue must produce a unique and dynamic cinematic decision
- DO NOT repeat the same camera, emotion, or movement patterns unless context demands it
- Make decisions like a real director (emotion-driven, story-driven, visually expressive)

--------------------------------------------------

INPUT:

You will receive a list of dialogues.

For EACH dialogue, generate one or more beats.

You are also allowed to insert additional cinematic beats (non-dialogue).

--------------------------------------------------

CINEMATIC ENHANCEMENT RULE (VERY IMPORTANT):

You are NOT limited to character dialogue shots.

You MUST also insert additional beats when appropriate:
- establishing shots
- environmental shots
- object insert shots
- reaction shots
- silence beats

These beats may NOT contain dialogue.

--------------------------------------------------

WHEN TO ADD EXTRA BEATS:

1. Scene Start:
→ ALWAYS consider an establishing shot (environment, mood)

2. Emotional Peaks:
→ Add reaction shots (listener face, silence, gestures)

3. Suspense / Fear:
→ Insert environmental or object shots (door, clock, shadows)

4. Scene Transitions:
→ Use wide shots or environment to reset pacing

5. After Important Dialogue:
→ Add pause beats (visual storytelling without dialogue)

--------------------------------------------------

OBJECT-FOCUSED SHOT RULE (IMPORTANT):

When a beat focuses on an object or environment:

- DO NOT use "NONE"
- Use the OBJECT NAME as the speaker

Examples:
- Clock → "CLOCK"
- Door → "DOOR"
- Wind → "WIND"
- Dust → "DUST"
- Watch → "WATCH"
- Environment → "ENVIRONMENT"

--------------------------------------------------

BEAT STRUCTURE:

Each beat must contain:

- scene_id
- beat_id
- speaker
- dialogue
- action
- emotion
- intent
- shot_type
- camera_angle
- camera_movement
- focus_target
- secondary_target
- duration
- transition

--------------------------------------------------

DIRECTOR LOGIC:

1. Emotion Detection (choose ONLY from allowed list):

neutral, calm, curious, fear, anger, tension, relief, sadness, shock

--------------------------------------------------

2. Intent Detection:

question, command, argument, explanation, reassure, confront, emphasize, establish

--------------------------------------------------

3. Shot Type:

- extreme_close_up → intense emotion or object detail
- close_up → strong emotional focus
- medium_close_up → emotional dialogue
- medium_shot → normal conversation
- two_shot → interaction
- over_shoulder → conversational perspective
- reaction_shot → listener response
- wide_shot → environment / isolation

--------------------------------------------------

4. Camera Angle:

- eye_level → normal
- low_angle → dominance
- high_angle → vulnerability
- dutch_angle → tension/confusion

--------------------------------------------------

5. Camera Movement:

- static → calm
- pan → dialogue exchange
- tilt → reveal
- dolly_in / push_in → emotional intensity
- dolly_out / pull_out → emotional distance
- handheld → tension

--------------------------------------------------

6. Duration Rules:

- Short dialogue → 1.5 – 2.5 sec
- Medium → 2.5 – 4 sec
- Emotional / pauses → 4 – 6 sec
- Establishing shot → 3 – 6 sec
- Insert shot → 1.5 – 3 sec

--------------------------------------------------

7. Focus Logic:

- focus_target = speaker
- secondary_target = other character if applicable, else null

--------------------------------------------------

8. Action Field:

- Describe visible action (character, object, or environment)
- Keep it short and cinematic

Examples:
- "Murph looks scared"
- "Clock ticks loudly"
- "Dust blows across the field"

--------------------------------------------------

9. Transition:

- Default → cut
- dissolve → emotional continuity
- fade → scene opening/ending
- match_cut → visual continuity

--------------------------------------------------

QUALITY REQUIREMENTS:

- Output must feel like a real movie scene
- Maintain cinematic rhythm (not just dialogue)
- Use environment and objects meaningfully
- Avoid repetitive patterns
- Emotion must influence camera decisions

--------------------------------------------------

OUTPUT FORMAT (STRICT JSON LIST):

[
  {
        "scene_id": <int>,
    "beat_id": <int>,
    "speaker": "<string>",
    "dialogue": "<string>",
    "action": "<string>",
    "emotion": "<string>",
    "intent": "<string>",
    "shot_type": "<string>",
    "camera_angle": "<string>",
    "camera_movement": "<string>",
    "focus_target": "<string>",
    "secondary_target": "<string or null>",
    "duration": <float>,
    "transition": "<string>"
  }
]

--------------------------------------------------

Now process the input dialogues and generate a cinematic Beat Script.
"""

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False
        )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        try:
            json_start = response.find("[")
            json_end = response.rfind("]") + 1
            return eval(response[json_start:json_end])
        except:
            return []