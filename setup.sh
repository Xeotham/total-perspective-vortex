#!/bin/sh

if [ ! -d ./.venv ]; then \
  echo "Creating Venv..."; \
  python3 -m venv .venv; \
  if [ ! -f ./requirements.txt ]; then \
    echo "requirements.txt not found...";\
  else \
    .venv/bin/pip install -r ./requirements.txt ; \
  fi;
  echo "Venv created." ; \
fi

. ./.venv/bin/activate
echo "Venv sourced."