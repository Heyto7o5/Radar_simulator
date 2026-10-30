#!/usr/bin/env python3
"""Launch and inspect project-local long jobs independently of terminal sessions."""
import argparse
import os
import subprocess
import sys
import time
import psutil

from radar_platform.navigation.io import ROOT,local_path,read_json,write_json


def inspect_job(path):
    record=read_json(path)
    try:
        process=psutil.Process(record['pid'])
        alive=abs(process.create_time()-record['process_create_time'])<.01 and process.status()!=psutil.STATUS_ZOMBIE
        command=process.cmdline() if alive else []
        alive=alive and command==record['command']
        return {**record,'alive':alive,'process_status':process.status() if alive else 'not_the_recorded_process'}
    except psutil.NoSuchProcess:return {**record,'alive':False,'process_status':'terminated'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='action',required=True)
    start=sub.add_parser('start');start.add_argument('--record',required=True)
    start.add_argument('--log',required=True);start.add_argument('command',nargs=argparse.REMAINDER)
    inspect=sub.add_parser('inspect');inspect.add_argument('--record',required=True)
    args=p.parse_args();record=local_path(args.record)
    if args.action=='inspect':
        print(__import__('json').dumps(inspect_job(record)));return
    if record.exists() and inspect_job(record)['alive']:raise RuntimeError('recorded job is still alive; duplicate launch refused')
    if not args.log or not args.command:raise ValueError('log and script command required')
    command=args.command[1:] if args.command[0]=='--' else args.command
    script=local_path(command[0])
    if script.suffix!='.py':raise ValueError('only a project-contained Python script may be launched')
    argv=[sys.executable,str(script),*command[1:]];log=local_path(args.log);log.parent.mkdir(parents=True,exist_ok=True)
    environment=dict(os.environ)
    environment.update(PYTHONNOUSERSITE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
        NUMBA_CACHE_DIR=str(ROOT/'.cache/navigation/numba'),MPLCONFIGDIR=str(ROOT/'.cache/navigation/matplotlib'),
        TMPDIR=str(ROOT/'.cache/navigation/tmp'))
    with log.open('a') as stream:
        process=subprocess.Popen(argv,cwd=ROOT,env=environment,stdin=subprocess.DEVNULL,
            stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    info=dict(pid=process.pid,process_create_time=psutil.Process(process.pid).create_time(),
        command=argv,log=str(log),started_unix_s=time.time())
    write_json(record,info);print(__import__('json').dumps(info))


if __name__=='__main__':main()
