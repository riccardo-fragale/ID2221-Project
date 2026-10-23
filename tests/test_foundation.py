"""Validate configuration failures and environment checks without a Spark JVM."""

from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from amazon_recommender.cli import main
from amazon_recommender.config import load_config
from amazon_recommender.environment import inspect_environment


ROOT = Path(__file__).resolve().parents[1]


class ConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.template = (ROOT / "configs/local.toml").read_text()

    def load_modified(self, before, after):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "settings.toml"
            path.write_text(self.template.replace(before, after))
            return load_config(path)

    def test_paths_resolve_from_config_location_not_working_directory(self):
        config = load_config(ROOT / "configs/local.toml")
        self.assertEqual(config.storage_root, ROOT / "storage")
        self.assertEqual(config.digest, load_config(ROOT / "configs/local.toml").digest)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "settings.toml"
            path.write_text(self.template.replace('"../storage"', '"generated"'))
            local = load_config(path)
            self.assertFalse(local.storage_root.exists())

    def test_rejects_unapproved_cluster_and_typographical_errors(self):
        for before, after in [
            ('profile = "local"', 'profile = "full"'),
            ('master = "local[2]"', 'master = "local[*]"'),
            ('driver_memory = "2g"', 'driver_memory = "0g"'),
            ('shuffle_partitions = 4', 'shuffle_partitions = true'),
            ('event_log_enabled = false', 'event_log_enabled = "false"'),
            ('seed = 2221', 'seed = -1'),
            ('uri_env = "MONGODB_URI"', 'uri_env = "mongodb://password"'),
            ('storage_root = "../storage"', 'storage_root = "s3://bucket"'),
            ('shuffle_partitions = 4', 'shuffle_partition = 4'),
            ('user_fraction = 1.0', 'user_fraction = 0'),
            ('split = [0.8, 0.1, 0.1]', 'split = [0.8, 0.2, 0.2]'),
            ('split = [0.8, 0.1, 0.1]', 'split = [nan, 0.1, 0.1]'),
            ('bias_tolerance = 0.0001', 'bias_tolerance = nan'),
            ('shortlist_cap = 1000', 'shortlist_cap = 10'),
        ]:
            with self.subTest(after=after), self.assertRaises(ValueError):
                self.load_modified(before, after)

    def test_digest_changes_with_resource_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "settings.toml"
            path.write_text(self.template)
            first = load_config(path)
            path.write_text(self.template.replace('master = "local[2]"', 'master = "local[3]"'))
            second = load_config(path)
        self.assertNotEqual(first.digest, second.digest)

    def test_invalid_cli_configuration_exits_cleanly(self):
        with redirect_stderr(StringIO()) as stream:
            code = main(["validate-config", "--config", "/nonexistent/id2221.toml"])
        self.assertEqual(code, 2)
        self.assertIn("Invalid configuration", stream.getvalue())

    def test_exploratory_split_is_flexible_but_final_protocol_is_enforced(self):
        exploratory = self.load_modified('split = [0.8, 0.1, 0.1]', 'split = [1.0, 0.0, 0.0]')
        self.assertEqual(exploratory.experiment.split, (1, 0, 0))
        final = self.load_modified('purpose = "exploratory"', 'purpose = "final"')
        self.assertEqual(final.experiment.purpose, "final")
        self.template = self.template.replace('purpose = "exploratory"', 'purpose = "final"')
        with self.assertRaisesRegex(ValueError, "final evaluation"):
            self.load_modified('split = [0.8, 0.1, 0.1]', 'split = [0.7, 0.2, 0.1]')


class EnvironmentTests(unittest.TestCase):
    @patch("amazon_recommender.environment.platform.python_version_tuple", return_value=("3", "11", "14"))
    @patch("amazon_recommender.environment.version", side_effect=lambda name: {"pyspark": "3.5.6", "numpy": "1.26.4"}[name])
    @patch("amazon_recommender.environment.subprocess.run")
    def test_java_home_has_priority_and_unsupported_java_is_rejected(self, run, *_):
        run.return_value = subprocess.CompletedProcess([], 0, "", 'openjdk version "25.0.0"')
        with patch.dict("os.environ", {"JAVA_HOME": "/chosen/jdk"}):
            report = inspect_environment()
        self.assertEqual(run.call_args.args[0], ["/chosen/jdk/bin/java", "-version"])
        self.assertTrue(any("Java 17" in message for message in report["errors"]))
        self.assertFalse(report["spark_started"])

    def test_config_command_does_not_import_pyspark(self):
        # Run in a fresh interpreter so another test cannot hide an eager import.
        script = (
            "import sys; from amazon_recommender.cli import main; "
            f"assert main(['validate-config', '--config', {str(ROOT / 'configs/local.toml')!r}]) == 0; "
            "assert 'pyspark' not in sys.modules"
        )
        result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
