#!/bin/bash

srun --gres=gpu:4 -c 24 --qos=m --time=04:00:00 --mem 48G -p a40 --pty bash
