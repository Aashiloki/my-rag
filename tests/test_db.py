import importlib
import unittest
from unittest.mock import patch


class DbClientLockTests(unittest.TestCase):
    def test_get_qdrant_client_handles_locked_local_storage(self):
        import app.db as db_module

        with patch.dict("os.environ", {"QDRANT_URL": "", "QDRANT_API_KEY": ""}, clear=False), patch.object(
            db_module, "QDRANT_PATH", "C:/tmp/qdrant-test"
        ), patch(
            "qdrant_client.QdrantClient",
            side_effect=RuntimeError("Storage folder ... is already accessed by another instance of Qdrant client"),
        ):
            reloaded = importlib.reload(db_module)
            self.assertIsNone(reloaded.client)
            self.assertIsNone(reloaded.get_qdrant_client())


if __name__ == "__main__":
    unittest.main()
