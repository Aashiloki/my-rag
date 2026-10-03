import unittest

import app.chat as chat


class ChatCacheTests(unittest.TestCase):
    def test_clear_cache_removes_stale_answers(self):
        chat._cache["stale-question"] = "old answer"
        chat.clear_cache()
        self.assertEqual(chat._cache, {})


if __name__ == "__main__":
    unittest.main()
