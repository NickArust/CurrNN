#!/bin/bash

find . -path './tensorboardenv' -prune -o -name '*.py' -print | xargs wc -l
find . -path './tensorboardenv' -prune -o -name '*.c' -print | xargs wc -l
find . -path './tensorboardenv' -prune -o -name '*.cpp' -print | xargs wc -l
find . -path './tensorboardenv' -prune -o -name '*.m' -print | xargs wc -l
find . -path './tensorboardenv' -prune -o -name '*.slurm' -print | xargs wc -l
find . -path './tensorboardenv' -prune -o -name '*.txt' -print | xargs wc -l
