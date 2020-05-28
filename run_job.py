#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import subprocess
import os
import sys
import shutil

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Utility to launch jobs on the cluster.")
    parser.add_argument('-r', '--results', required=True,
            help="Path to the results directory")
    parser.add_argument('--interactive', default=False, action='store_true',
            help="Run the job in interactive mode")
    parser.add_argument('--resume', default=False, action='store_true',
            help="Resume the job with the source in the results dir.")
    parser.add_argument('--batch32', default=False, action='store_true',
            help='Run the job in the batch_32GB partition for more GPU memory.')
    parser.add_argument('--gpus', default=8, type=int, help='The number of GPUs to allocate')

    # Get the results directory without fussing with the rest
    args, rest_args = parser.parse_known_args()

    results_dir = args.results

    print("Creating results dir...")
    os.makedirs(results_dir, exist_ok=True)
    print("Done")

    source_dir = os.path.dirname(__file__)
    print('Source directory:', source_dir)

    dest_dir = os.path.join(results_dir, 'source')

    if args.resume:
    assert os.path.exists(dest_dir)
    else:
    if os.path.exists(dest_dir):
        shutil.rmtree(dest_dir)

    print('Copying source code to "{}"...'.format(dest_dir))
    shutil.copytree(source_dir, dest_dir, symlinks=True,
        ignore=shutil.ignore_patterns('synthetic_data', 'logs', '.git*',
                    '__pycache__', '.vscode')
    )
    print('Done')

    proc = subprocess.run(['bash', 'git_branch.sh'], stdout=subprocess.PIPE)

    git_info = proc.stdout.decode('utf-8').strip()

    git_dir = os.path.join(dest_dir, 'git-info')
    os.makedirs(git_dir, exist_ok=True)

    with open(os.path.join(git_dir, 'revision.txt'), 'w') as fd:
        fd.write(git_info)
        fd.write('\n')
        print('Wrote git revision info to "{}".'.format(fd.name))

    proc = subprocess.run(['git', '--no-pager', 'diff'], stdout=subprocess.PIPE)

    git_diff = proc.stdout.decode('utf-8').strip()

    with open(os.path.join(git_dir, 'delta.diff'), 'w') as fd:
        fd.write(git_diff)
        fd.write('\n')
        print('Wrote git diff to "{}".'.format(fd.name))


    command_args = rest_args + ['--results', results_dir]

    if args.resume:
    command_args = command_args + ['--resume']

    command_args = subprocess.list2cmdline(command_args)
    print('command: {}'.format(command_args))

    with open('docker_image', 'r') as fd:
    docker_image = fd.read().strip()

    print('docker image: {}'.format(docker_image))

    submit_args = [
    'submit_job',
    '--cpu', str(10 * args.gpus),
    '--gpu', str(args.gpus),
    '--mounts', os.environ['MOUNTS'],
    '--workdir', dest_dir,
    '--image', docker_image,
    '--pre_timeout_signal', '60',
    '--coolname',
    '-c', command_args
    ]

    if args.interactive:
    submit_args.insert(1, '--interactive')

    if args.batch32:
    submit_args.insert(1, '--partition')
    submit_args.insert(2, 'batch_32GB')

    print('submit_args:', submit_args)

    popen = subprocess.Popen(submit_args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                 universal_newlines=True)
    for stdout_line in iter(popen.stdout.readline, ""):
    print(stdout_line, end="")
    popen.stdout.close()
    return_code = popen.wait()

    sys.exit(return_code)
