#!/bin/sh
swarm -f restroke_bs_all_0.swarm -g 100 -t 6 --module python/3.6 --gres=gpu:p100:2 --partition=gpu
swarm -f restroke_b_all_0.swarm -g 100 -t 6 --module python/3.6 --gres=gpu:p100:2 --partition=gpu
swarm -f restroke_c_all_0.swarm -g 100 -t 6 --module python/3.6 --gres=gpu:p100:2 --partition=gpu