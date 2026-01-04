#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse

# Add project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from opencontext.config.global_config import GlobalConfig
from opencontext.storage.global_storage import GlobalStorage
from opencontext.context_processing.merger.context_merger import ContextMerger
from opencontext.utils.logging_utils import setup_logging, get_logger

logger = get_logger(__name__)

def run_cleanup(config_path: str = None, **kwargs):
    """
    Run the memory cleanup task.
    
    Args:
        config_path: Path to the configuration file.
        **kwargs: Additional arguments (ignored for now, but kept for compatibility).
    """
    if not config_path:
        config_path = os.path.join(project_root, "config", "config.yaml")
        
    logger.info(f"Initializing with config: {config_path}")
    
    # Initialize GlobalConfig
    GlobalConfig.get_instance().initialize(config_path)
    config = GlobalConfig.get_instance().get_config()
    
    # Initialize logging (allow caller to disable re-init to avoid mixing)
    init_logging = kwargs.get("init_logging", True)
    if init_logging:
        setup_logging(config.get("logging", {}))
    
    # Initialize GlobalStorage (auto-init relies on GlobalConfig being ready)
    storage = GlobalStorage.get_instance().get_storage()
    if not storage:
        logger.error("Failed to initialize storage. Exiting.")
        return
        
    logger.info("Storage initialized successfully.")
    
    # Initialize ContextMerger
    try:
        merger = ContextMerger()
        logger.info("ContextMerger initialized. Starting cleanup...")
        
        # Run cleanup
        merger.intelligent_memory_cleanup()
        logger.info("Cleanup task finished.")
        
    except Exception as e:
        logger.error(f"Error during cleanup task: {e}", exc_info=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run memory cleanup task")
    parser.add_argument("--config", help="Path to configuration file", default=None)
    args = parser.parse_args()
    
    run_cleanup(config_path=args.config)
