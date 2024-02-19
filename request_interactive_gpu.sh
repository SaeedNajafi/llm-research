#!/bin/bash

srun --gres=gpu:1 -c 8 --qos=m2 --time=08:00:00 --mem 16G -p a40 --pty bash
