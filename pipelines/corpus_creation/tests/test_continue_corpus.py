import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import continue_corpus as c


class Client:
    def __init__(self, fail_teacher=False, complete=True):
        self.chat = SimpleNamespace(completions=self)
        self.calls = []
        self.fail_teacher = fail_teacher
        self.complete = complete

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if "response_format" in kwargs:
            if self.fail_teacher:
                raise RuntimeError("interrupted")
            text = json.dumps({"thought_process": "check", "student_emotion": "Engaged",
                "roadmap_breakdown": "steps", "next_step_plan": "done", "is_completed": self.complete,
                "teacher_utterance": "Good."})
        else:
            text = "I think it is 4."
        return SimpleNamespace(id=f"response-{len(self.calls)}", model=kwargs["model"],
                               usage=None, choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content=text))])


class Tests(unittest.TestCase):
    def test_math_delimiters_are_not_mistaken_for_latex_commands(self):
        self.assertEqual(c.repair_surface("\x08r^2\x08"), "r^2")
        self.assertEqual(c.repair_surface(r"\br^2\b"), "r^2")
        self.assertEqual(c.repair_surface("r,s\neq0"), r"r,s\neq0")
        self.assertEqual(c.repair_surface(r"\boxed{1}+\beta"), r"\boxed{1}+\beta")
        self.assertEqual(c.repair_surface(r"\arg z,\varnothing,\bf x,\bm y"), r"\arg z,\varnothing,\bf x,\bm y")

    def test_recovers_json_decoded_latex_controls(self):
        class EscapedClient(Client):
            def create(self, **kwargs):
                response = super().create(**kwargs)
                value = json.loads(response.choices[0].message.content)
                value["teacher_utterance"] = "Check " + "\x1c" + "$" + "\x0c" + "rac{1}{2}$" + "\x1d" + "."
                response.choices[0].message.content = json.dumps(value)
                return response
        value, metadata = c.completion(EscapedClient(), [], "teacher", 0.2, True, {"max_completion_tokens": 100})
        self.assertEqual(value["teacher_utterance"], r"Check $\frac{1}{2}$.")
        self.assertIn("teacher_utterance", metadata["surface_repairs"])

    def test_resume_pending_student_does_not_repeat_student_api(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "state.json"
            config = {"max_turns": 15, "max_completion_tokens": 4096, "run_fingerprint": "fixed",
                      "student_model": "student", "teacher_model": "teacher"}
            question = {"id": "math_train_1098", "translated_question": "2+2?"}
            profiles = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                c.generate_one(Client(fail_teacher=True), question, 800, profiles, config, path)
            state = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(len(state["conversation"]), 1)
            resumed = Client()
            result = c.generate_one(resumed, question, 800, profiles, config, path)
            self.assertEqual(len(resumed.calls), 1)
            self.assertIn("response_format", resumed.calls[0])
            self.assertEqual(result["student_profile"]["id"], "c")
            self.assertEqual(result["generation_metadata"]["dialogue_number"], 801)
            self.assertTrue(result["is_completed"])
            self.assertEqual(len(result["conversation"]), 2)
            cached = Client()
            self.assertEqual(c.generate_one(cached, question, 800, profiles, config, path), result)
            self.assertEqual(len(cached.calls), 0)

    def test_max_turns_is_valid_incompleteness_not_api_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            config = {"max_turns": 2, "max_completion_tokens": 100, "run_fingerprint": "fixed",
                      "student_model": "student", "teacher_model": "teacher"}
            result = c.generate_one(Client(complete=False), {"id": "a", "translated_question": "Q"},
                                    800, [{}], config, Path(folder) / "state.json")
            self.assertFalse(result["is_completed"])
            self.assertEqual(len(result["conversation"]), 4)
            c.validate_state(result, 2, final=True)


if __name__ == "__main__":
    unittest.main()
