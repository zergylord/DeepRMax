#!/bin/bash
args="-use_egreedy
      -learn_start 1025
        "
th train.lua $args
