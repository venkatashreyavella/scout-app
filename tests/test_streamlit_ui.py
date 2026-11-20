import unittest
from pathlib import Path
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "streamlit_app.py")

def app():
    at = AppTest.from_file(APP, default_timeout=40)
    at.secrets.update({"OPENAI_API_KEY": "", "TAVILY_API_KEY": "", "AGENT_ACCESS_TOKEN": "", "OPENAI_MODEL": "gpt-4o-mini"})
    return at.run()

def button(at, label):
    return next(b for b in at.button if b.label == label)

class StreamlitInterfaceTests(unittest.TestCase):
    def test_default_question_runs_without_api_keys(self):
        at = app()
        self.assertEqual(len(at.exception), 0)
        button(at, "Run research").click().run()
        self.assertEqual(len(at.exception), 0)
        report = at.session_state["report"]
        self.assertEqual(report.result.status, "complete")
        self.assertEqual(report.result.calculations[0].value, 300)
        self.assertTrue(any(e.kind == "tool_end" for e in report.trace))

    def test_missing_evidence_example_renders_safe_result(self):
        at = app()
        button(at, "Missing evidence").click().run()
        button(at, "Run research").click().run()
        self.assertEqual(len(at.exception), 0)
        self.assertEqual(at.session_state["report"].result.status, "insufficient_evidence")

    def test_live_without_configuration_renders_error_without_crashing(self):
        at = app()
        at.radio[0].set_value("Live").run()
        button(at, "Run research").click().run()
        self.assertEqual(len(at.exception), 0)
        self.assertEqual(at.session_state["report"].result.status, "error")

if __name__ == "__main__":
    unittest.main()
