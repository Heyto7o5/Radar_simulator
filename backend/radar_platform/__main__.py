"""Installed CLI; no arguments preserves the API startup shortcut."""
import os
import sys
from .cli import main

if __name__ == '__main__':
    arguments = sys.argv[1:]
    if not arguments:
        arguments = ['serve']
        if os.environ.get('RADAR_RELOAD') == '1':
            arguments.append('--reload')
    main(arguments)
