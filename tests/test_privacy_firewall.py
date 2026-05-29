from __future__ import annotations

import unittest
from unittest import mock

from core.research import FakeWebBackend, PrivacyFirewall, ResearchPipeline, ResearchSource


class PrivacyFirewallTests(unittest.TestCase):
    def test_project_code_is_blocked_from_outbound_search(self):
        backend = FakeWebBackend()
        query = """Найди актуальную причину ошибки:
```python
def private_business_logic(secret):
    return secret + 1
```"""

        result = ResearchPipeline(backend).run(query)

        self.assertEqual(backend.search_calls, [])
        self.assertEqual(result.error, "privacy blocked")
        self.assertIn("code", result.privacy_reasons)

    def test_local_windows_paths_are_stripped_before_search(self):
        backend = FakeWebBackend([ResearchSource("Doc", "https://example.com")])
        query = r"Какая актуальная ошибка PermissionError в C:\Users\rebko\Private\main.py?"

        result = ResearchPipeline(backend).run(query)

        self.assertTrue(result.used_search)
        sent_query = backend.search_calls[0][0]
        self.assertNotIn(r"C:\Users", sent_query)
        self.assertIn("[local path]", sent_query)

    def test_api_key_like_strings_are_blocked(self):
        backend = FakeWebBackend()
        query = "Какая актуальная ошибка OpenAI API key sk-1234567890abcdefXYZ?"

        result = ResearchPipeline(backend).run(query)

        self.assertEqual(backend.search_calls, [])
        self.assertEqual(result.error, "privacy blocked")
        self.assertIn("secret", result.privacy_reasons)

    def test_chat_history_is_not_included_in_query(self):
        backend = FakeWebBackend([ResearchSource("Doc", "https://example.com")])
        query = "User: мой приватный чат\nAssistant: ответ\nКакая актуальная версия Python?"

        result = ResearchPipeline(backend).run(query, confirmed_outbound=True)

        sent_query = backend.search_calls[0][0]
        self.assertTrue(result.used_search)
        self.assertNotIn("User:", sent_query)
        self.assertNotIn("Assistant:", sent_query)
        self.assertNotIn("мой приватный чат", sent_query)

    def test_companion_memory_is_not_sent_without_confirmation(self):
        backend = FakeWebBackend()
        query = "companion_memory: user email rebko@example.com\nНайди актуальные новости Python"

        result = ResearchPipeline(backend).run(query)

        self.assertEqual(backend.search_calls, [])
        self.assertEqual(result.error, "privacy confirmation required")
        self.assertIn("companion_memory", result.privacy_reasons)

    def test_traceback_is_sanitized_to_error_type_and_message(self):
        backend = FakeWebBackend([ResearchSource("Doc", "https://example.com")])
        query = r"""Найди актуальное решение:
Traceback (most recent call last):
  File "C:\Users\rebko\secret_project\main.py", line 1, in <module>
    print(1 / 0)
ZeroDivisionError: division by zero
"""

        result = ResearchPipeline(backend).run(query)

        self.assertTrue(result.used_search)
        sent_query = backend.search_calls[0][0]
        self.assertEqual(sent_query, "Python ZeroDivisionError division by zero")
        self.assertNotIn("rebko", sent_query)
        self.assertNotIn("secret_project", sent_query)

    def test_local_path_with_error_after_path_becomes_generic_error_query(self):
        backend = FakeWebBackend([ResearchSource("Doc", "https://example.com")], {
            "https://example.com": "ZeroDivisionError division by zero explanation"
        })
        query = r"Найди ошибку D:\Zen Ai Editor\core\app_data.py ZeroDivisionError division by zero"

        result = ResearchPipeline(backend).run(query)

        self.assertEqual(backend.search_calls[0][0], "Python ZeroDivisionError division by zero")
        self.assertNotIn("Zen Ai Editor", backend.search_calls[0][0])

    def test_safe_generic_query_is_allowed(self):
        backend = FakeWebBackend([ResearchSource("Python", "https://www.python.org/")])

        result = ResearchPipeline(backend).run("Какая сейчас актуальная версия Python?")

        self.assertTrue(result.used_search)
        self.assertEqual(backend.search_calls[0][0], "Какая сейчас актуальная версия Python?")

    def test_sensitive_query_requires_confirmation(self):
        backend = FakeWebBackend()

        result = ResearchPipeline(backend).run("Найди актуальные утечки для user@example.com")

        self.assertEqual(result.error, "privacy confirmation required")
        self.assertEqual(backend.search_calls, [])
        self.assertIn("personal_data", result.privacy_reasons)

    def test_user_approved_explicit_outbound_payload_is_allowed_after_confirmation(self):
        backend = FakeWebBackend([ResearchSource("Doc", "https://example.com")])

        blocked = ResearchPipeline(backend).run("Найди актуальные данные для user@example.com")
        allowed = ResearchPipeline(backend).run(
            "Найди актуальные данные для user@example.com",
            confirmed_outbound=True,
        )

        self.assertEqual(blocked.error, "privacy confirmation required")
        self.assertTrue(allowed.used_search)
        self.assertEqual(backend.search_calls[0][0], "Найди актуальные данные для [personal email]")

    def test_logs_show_sanitized_query_not_private_raw_content(self):
        backend = FakeWebBackend()
        query = r"Какая актуальная ошибка в C:\Users\rebko\app.py с token=supersecret123?"

        with mock.patch("core.research.write_log") as log:
            ResearchPipeline(backend).run(query)

        rendered = "\n".join(str(call.args[0]) for call in log.call_args_list)
        self.assertIn("[research_query_sanitized]", rendered)
        self.assertNotIn(r"C:\Users\rebko", rendered)
        self.assertNotIn("supersecret123", rendered)
        self.assertIn("[secret]", rendered)

    def test_firewall_detection_helpers(self):
        firewall = PrivacyFirewall()

        self.assertTrue(firewall.detect_local_paths(r"C:\Users\rebko\app.py"))
        self.assertTrue(firewall.detect_secrets("api_key=abcdef123456"))
        self.assertTrue(firewall.detect_code_blocks("```python\nprint('x')\n```"))
        self.assertTrue(firewall.detect_personal_data_light("user@example.com"))

    def test_blocked_query_cannot_be_confirmed(self):
        backend = FakeWebBackend()
        result = ResearchPipeline(backend).run(
            "Найди ошибку\n```python\nSECRET_KEY='abc123'\ndef broken(): pass\n```",
            confirmed_outbound=True,
        )

        self.assertEqual(result.error, "privacy blocked")
        self.assertEqual(backend.search_calls, [])


if __name__ == "__main__":
    unittest.main()
