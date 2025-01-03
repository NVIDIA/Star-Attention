# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
from typing import List, Optional

from babilong import PROMPT_TEMPLATES

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TASKS = ['qa1', 'qa2', 'qa3', 'qa4', 'qa5']
SPLIT_NAMES = ['16k', '32k', '64k', '128k']


def define_cmd_arguments():
    parser = argparse.ArgumentParser()

    # Model Parameters
    parser.add_argument('-n', '--model_name', required=True, help='experiment name prefix')
    parser.add_argument('-p', '--model_path', required=True, help='model path')
    parser.add_argument(
        '-pc',
        '--prompt_config',
        required=True,
        choices=PROMPT_TEMPLATES.keys(),
        help='prompt template config name. options from `babilong/template.py`',
    )

    # Attention Configuration
    parser.add_argument('-a', '--attn_type', default='star', help='attention type')
    parser.add_argument('-bs', '--block_size', type=int, default=-1, help='context block size')
    parser.add_argument('-as', '--anchor_block_size', type=int, default=-1, help='anchor block size')

    # Sequence Lengths and Tasks
    parser.add_argument(
        '-l',
        '--seq_lengths',
        type=int,
        required=True,
        nargs='+',
        help='sequence lengths',
    )
    parser.add_argument('-t', '--tasks', default=TASKS, nargs='+', choices=TASKS, help='tasks')
    parser.add_argument(
        '-d', '--pregen_data_dir', default=None, help='name pre-generated data directory in the `dataset` folder'
    )

    # Distributed Inference Parameters
    parser.add_argument(
        '-nn', '--num_nodes', type=int, default=1, help='number of nodes. For dense attention, default is set to 1.'
    )
    parser.add_argument('-np', '--nproc_per_node', type=int, default=None, help='number of processes per node')

    # Logging
    parser.add_argument(
        '--output_dir',
        default=os.path.join(BASE_DIR, 'results'),
        help='results directory',
    )

    return parser.parse_args()


def submit_job(cmd, log_dir, filename):
    # Save the command to a file
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, filename), 'w') as f:
        f.write(cmd)

    # Submit the job
    os.system(f'cd {BASE_DIR}; {cmd}')


def main(
    model_path: str,
    attn_type: str,
    block_size: int,
    anchor_block_size: int,
    prompt_config: str,
    seq_lengths: List[int],
    tasks: List[str],
    nproc_per_node: int,
    output_dir: str,
    num_nodes: int = 1,
    pregen_data_dir: Optional[str] = None,
):
    if 'star' in attn_type:
        assert (
            block_size >= anchor_block_size
        ), f'block_size ({block_size}) must be greater than anchor_block_size ({anchor_block_size})'

    # Path to any pre-generated data, if exists
    if pregen_data_dir is not None:
        pregen_data_dir = os.path.join(BASE_DIR, 'dataset', pregen_data_dir)

    # Inference Parameters
    stop_words = ','.join(PROMPT_TEMPLATES[prompt_config]['stop_words'])

    # Schedule jobs for each sequence length
    for seq_length in seq_lengths:
        seq_length_repr = f'{seq_length // 1024}k'
        if 'star' in attn_type and block_size + anchor_block_size > seq_length:
            print(
                f'block_size + anchor_block_size ({block_size + anchor_block_size}) '
                f'must be less than or equal to seq_length ({seq_length}). '
                'Skipping...'
            )
            continue

        # Depending on the sequence length and the block size, adjust the number of processes
        if attn_type != 'dense':
            nproc_per_node_seq_len = min(nproc_per_node, seq_length // block_size)
            inference_executor = f'torchrun --nnodes={num_nodes} --nproc_per_node={nproc_per_node_seq_len}'
        else:
            inference_executor = 'python'

        results_dir = os.path.join(output_dir, seq_length_repr)
        log_dir = os.path.join(results_dir, 'logs')
        data_dir = os.path.join(results_dir, 'data')
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(data_dir, exist_ok=True)

        # Evaluate each task
        for task in tasks:
            task_log_dir = os.path.join(log_dir, task)

            # Prepare dataset
            task_data_file = (
                os.path.join(pregen_data_dir, seq_length_repr, f'{task}_{seq_length_repr}.jsonl')
                if pregen_data_dir
                else None
            )
            if task_data_file is None or not os.path.exists(task_data_file):
                data_gen_cmd = (
                    f'python babilong/prepare_data.py '
                    f'--download_dir {os.path.join(BASE_DIR, "dataset", "babilong")} '
                    f'--output_dir {data_dir} '
                    f'--tasks {task} '
                    f'--split_names {seq_length_repr} '
                    f'--model_template_type {prompt_config} '
                )
                submit_job(data_gen_cmd, task_log_dir, f'data_generation.sh')
                task_data_file = os.path.join(data_dir, f'{task}_{seq_length_repr}.jsonl')

            # Run response generation
            task_gen_cmd = (
                f'{inference_executor} run_star_attn_inference.py '
                f'--model_path {model_path} '
                f'--attn_type {attn_type} '
                f'--block_size {block_size} '
                f'--anchor_block_size {anchor_block_size} '
                f'--tokens_to_generate 128 '
                f'--input_path {task_data_file} '
                f'--output_path {os.path.join(results_dir, task)}.jsonl'
                f'--stop_words {stop_words}'
            )
            submit_job(task_gen_cmd, task_log_dir, 'generate_predictions.sh')


if __name__ == '__main__':
    # Parse command line arguments
    args = define_cmd_arguments()

    # Validate sequence lengths
    for seq_length in args.seq_lengths:
        assert f'{seq_length // 1024}k' in SPLIT_NAMES, f'seq_length must be one of {SPLIT_NAMES}'

    # Validate star attention parameters
    if 'star' in args.attn_type:
        assert args.block_size > 0, 'block_size must be greater than 0'

    # Validate star and ring attention parameters
    if args.attn_type != 'dense':
        assert args.nproc_per_node is not None and args.nproc_per_node > 0, 'nproc_per_node must be greater than 0'
        assert args.num_nodes > 0, 'num_nodes must be greater than 0'

    # Validate model path
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f'{args.model_path} not found')

    # Setup the model name and output directory
    model_name_suffix = ''
    if 'star' in args.attn_type:
        anchor_block_size = args.anchor_block_size if args.anchor_block_size > 0 else args.block_size
        anchor_block_size_repr = (
            f'{anchor_block_size}' if anchor_block_size < 1024 else f'{anchor_block_size // 1024}k'
        )
        block_size_repr = f'{args.block_size}' if args.block_size < 1024 else f'{args.block_size // 1024}k'
        model_name_suffix = f'_b{block_size_repr}a{anchor_block_size_repr}'
    args.output_dir = os.path.join(args.output_dir, f'{args.model_name}_{args.attn_type}{model_name_suffix}')

    main(
        args.model_path,
        args.attn_type,
        args.block_size,
        args.anchor_block_size,
        args.prompt_config,
        args.seq_lengths,
        args.tasks,
        args.nproc_per_node,
        args.output_dir,
        num_nodes=args.num_nodes,
        pregen_data_dir=args.pregen_data_dir,
    )
