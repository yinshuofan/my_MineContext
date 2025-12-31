#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import yaml
import time
import importlib.util
import argparse
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.date import DateTrigger

# Add project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from opencontext.utils.logging_utils import setup_logging, get_logger

# Configure basic logging for the scheduler itself
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SchedulerRunner")

def load_task_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def get_trigger(trigger_type, trigger_args):
    if trigger_type == 'interval':
        return IntervalTrigger(**trigger_args)
    elif trigger_type == 'cron':
        return CronTrigger(**trigger_args)
    elif trigger_type == 'date':
        return DateTrigger(**trigger_args)
    else:
        raise ValueError(f"Unknown trigger type: {trigger_type}")

def get_task_module(script_path):
    script_name = os.path.basename(script_path).replace('.py', '')
    spec = importlib.util.spec_from_file_location(script_name, script_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[script_name] = module
    spec.loader.exec_module(module)
    return module

def job_wrapper(func, **kwargs):
    try:
        logger.info(f"Starting job: {func.__name__}")
        func(**kwargs)
        logger.info(f"Job finished: {func.__name__}")
    except Exception as e:
        logger.error(f"Job failed: {func.__name__}, Error: {e}", exc_info=True)

def main():
    parser = argparse.ArgumentParser(description="Unified Task Scheduler Runner")
    parser.add_argument("--config", required=True, help="Path to tasks.yaml")
    parser.add_argument("--project-config", help="Path to project config (passed to tasks)", default=None)
    args = parser.parse_args()

    if not os.path.exists(args.config):
        logger.error(f"Config file not found: {args.config}")
        return

    task_config = load_task_config(args.config)
    tasks = task_config.get('tasks', {})

    scheduler = BackgroundScheduler()

    for task_name, task_info in tasks.items():
        if not task_info.get('enabled', True):
            continue

        script_file = task_info.get('script')
        entry_point = task_info.get('entry_point', 'main')
        trigger_type = task_info.get('trigger', 'interval')
        trigger_args = task_info.get('trigger_args', {})
        task_args = task_info.get('args', {})
        
        # Pass project config if available and not overridden
        if args.project_config and 'config_path' not in task_args:
            task_args['config_path'] = args.project_config

        # Resolve script path relative to scripts directory
        if not os.path.isabs(script_file):
            script_path = os.path.join(project_root, "scripts", script_file)
        else:
            script_path = script_file

        if not os.path.exists(script_path):
            logger.error(f"Script not found for task {task_name}: {script_path}")
            continue

        try:
            logger.info(f"Loading task: {task_name} from {script_path}")
            module = get_task_module(script_path)
            func = getattr(module, entry_point)
            
            trigger = get_trigger(trigger_type, trigger_args)
            
            scheduler.add_job(
                job_wrapper,
                trigger=trigger,
                args=[func],
                kwargs=task_args,
                id=task_name,
                name=task_name,
                replace_existing=True
            )
            logger.info(f"Scheduled task: {task_name} with trigger: {trigger_type} {trigger_args}")
            
        except Exception as e:
            logger.error(f"Failed to load/schedule task {task_name}: {e}", exc_info=True)

    if not scheduler.get_jobs():
        logger.warning("No jobs scheduled. Exiting.")
        return

    scheduler.start()
    logger.info("Scheduler started. Press Ctrl+C to exit.")

    try:
        while True:
            time.sleep(1)
    except (KeyboardInterrupt, SystemExit):
        logger.info("Stopping scheduler...")
        scheduler.shutdown()

if __name__ == "__main__":
    main()
