.PHONY: all

# A standard makefile for a nodejs project

all:
	npm run build
	cd examples/sgemm && npm run build

watch:
	fd '.*\.ts' | entr -c make