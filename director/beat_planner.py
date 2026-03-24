from ai.local_llm import LocalLLM


class BeatPlanner:

    def __init__(self):
        self.llm = LocalLLM()

    def plan_beats(self, scene):

        beats = []
        beat_id = 1
        current_time = 0.0

        dialogue_units = scene["dialogue_units"]

        # 🔥 ONE LLM CALL (not loop)
        results = self.llm.analyze_scene(dialogue_units)

        for i, unit in enumerate(dialogue_units):

            result = results[i] if i < len(results) else {}

            duration = float(result.get("duration", 2.5))

            beat = {
                "scene_id": scene.get("scene_id", 1),
                "beat_id": beat_id,
                "speaker": unit["speaker"],
                "dialogue": unit["dialogue"],

                "emotion": result.get("emotion", "neutral"),
                "intent": result.get("intent", "statement"),
                "shot_type": result.get("shot_type", "medium"),
                "camera_angle": result.get("camera_angle", "eye_level"),
                "camera_movement": result.get("camera_movement", "static"),

                "start_time": round(current_time, 2),
                "duration": round(duration, 2)
            }

            beats.append(beat)

            current_time += duration
            beat_id += 1

        return beats