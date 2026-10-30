#!/usr/bin/env python3
"""Execute a causal physical radar navigation episode."""
import argparse
import json
from radar_platform.navigation.io import read_json,write_json,local_path
from radar_platform.navigation.episode import default_episode,run_episode


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True);p.add_argument('--model-dir')
    p.add_argument('--spec');p.add_argument('--mode',choices=['learned','oracle','no_correction'],default='learned')
    p.add_argument('--threads',type=int,default=4);p.add_argument('--maximum-frames',type=int)
    p.add_argument('--filter-state',action='store_true')
    p.add_argument('--gentle-terminal',action='store_true')
    args=p.parse_args();spec=read_json(args.spec) if args.spec else default_episode()
    try:
        result=run_episode(spec,args.output,mode=args.mode,model_dir=args.model_dir,
            threads=args.threads,maximum_frames=args.maximum_frames,state_filter=args.filter_state,gentle_terminal=args.gentle_terminal)
    except Exception as exc:
        root=local_path(args.output)
        if root.exists() and not (root/'complete.json').exists():
            previous=read_json(root/'status.json') if (root/'status.json').exists() else None
            write_json(root/'failure.json',dict(kind='execution_error',exception=type(exc).__name__,message=str(exc),last_status=previous))
            write_json(root/'status.json',dict(status='failed',exception=type(exc).__name__,message=str(exc)))
        raise
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
