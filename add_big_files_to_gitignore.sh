#!/bin/bash
# Written by: samhaug
# 2021 May 17 06:45:38 PM

find . -size +100M | cat >> .gitignore

