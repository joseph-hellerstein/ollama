#!/bin/bash
source ../BaseStack/bin/setup_run.sh
PYTHONPATH=`pwd`/src:${PYTHONPATH}:../controlSBML/src
export PYTHONPATH
MYPYPATH=`pwd`/controlSBML/src
export MYPYPATH
source oll/bin/activate
