"""
Configuration module
"""

import json
import os

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

        # Write configuration
        with open(f"{path}/{name}", "w" if jsonconfig else "wb", encoding="utf-8" if jsonconfig else None) as handle:
            if jsonconfig:
                # Write config as JSON
                json.dump(config, handle, default=str, indent=2)
            else:
                # Backwards compatible method to save pickle configuration
                SerializeFactory.create("pickle").savestream(config, handle)
