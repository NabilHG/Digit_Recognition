#!/bin/bash

if [ $# -ne 2  ]; then
	echo "First parameter should be the file to be compile. The second one, the output file. Aborting compilation."
else
	g++ $1 -o $2
fi
