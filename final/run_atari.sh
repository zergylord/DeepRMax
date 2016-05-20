#!/bin/bash
args="-use_rmax
      -learn_start 33
        "
th train.lua $args
