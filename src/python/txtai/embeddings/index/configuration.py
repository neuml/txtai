"""
Configuration module
"""

import json
import os
import shutil

from tempfile import TemporaryDirectory

from ...serialize import SerializeFactory


class Configuration:
    """
    Loads and saves index configuration.
    """

    def load(self, path):
        """
        Loads index configuration. This method supports both config.json and config pickle files.

        Args:
            path: path to directory

        Returns:
            dict
        """

        # Configuration
        config = None

        # Determine if config is json or pickle
        jsonconfig = os.path.exists(f"{path}/config.json")

        # Set config file name
        name = "config.json" if jsonconfig else "config"

        # Load configuration
        with open(f"{path}/{name}", "r" if jsonconfig else "rb", encoding="utf-8" if jsonconfig else None) as handle:
            # Load JSON, also backwards-compatible with pickle configuration
            config = json.load(handle) if jsonconfig else SerializeFactory.create("pickle").loadstream(handle)

        # Add format parameter
        config["format"] = "json" if jsonconfig else "pickle"

        return config

    def save(self, config, path):
        """
        Saves index configuration. This method defaults to JSON and falls back to pickle.

        Args:
            config: configuration to save
            path: path to directory

        Returns:
            dict
        """

        # Default to JSON config
        jsonconfig = config.get("format", "json") == "json"

        # Set config file name
        name = "config.json" if jsonconfig else "config"

        # Keep the last complete configuration readable until the new file is closed.
        destination = os.path.join(path, name)
        with TemporaryDirectory(dir=path) as directory:
            staged = os.path.join(directory, name)
            with open(staged, "w" if jsonconfig else "wb", encoding="utf-8" if jsonconfig else None) as handle:
                if jsonconfig:
                    json.dump(config, handle, default=str, indent=2)
                else:
                    SerializeFactory.create("pickle").savestream(config, handle)

            if os.path.exists(destination):
                shutil.copymode(destination, staged)
            os.replace(staged, destination)
