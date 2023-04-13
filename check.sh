#!/bin/bash

set -ex

EXIT_STATUS=0

# Autoformatter *first*, to avoid double-reporting errors
if ! black --check setup.py stackscope; then
    EXIT_STATUS=1
    black --diff setup.py stackscope
fi

# Run flake8 without pycodestyle and import-related errors
flake8 stackscope/ \
    --ignore=D,E,W,F401,F403,F405,F821,F822\
    || EXIT_STATUS=$?

# Run mypy
mypy --strict -p stackscope || EXIT_STATUS=$?

# Finally, leave a really clear warning of any issues and exit
if [ $EXIT_STATUS -ne 0 ]; then
    cat <<EOF
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

Problems were found by static analysis (listed above).
To fix formatting and see remaining errors, run

    pip install -r test-requirements.txt
    black setup.py stackscope
    ./check.sh

in your local checkout.

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
EOF
    exit 1
fi
exit 0
