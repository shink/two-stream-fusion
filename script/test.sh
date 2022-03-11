#!/bin/bash

python test/test.py

if [ -d log ] && [ -f log/c3d.log ]
then
  cat log/c3d.log
fi
