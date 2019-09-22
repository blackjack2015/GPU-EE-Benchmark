# Makefile: This file is part of the gpumembench micro-benchmark suite.
# 
# Contact: Elias Konstantinidis <ekondis@gmail.com>

.PHONY: all clean rebuild

all:
	$(MAKE) -C A4-dram
	$(MAKE) -C A5-cache
	$(MAKE) -C A6-shared
	$(MAKE) -C A7-texture
	mkdir -p ../linux
	cp A4-dram/A4-dram A5-cache/A5-cache A6-shared/A6-shared A7-texture/A7-texture ../linux/

clean:
	$(MAKE) -C A4-dram clean
	$(MAKE) -C A5-cache clean
	$(MAKE) -C A6-shared clean
	$(MAKE) -C A7-texture clean

rebuild:
	$(MAKE) -C A4-dram rebuild
	$(MAKE) -C A5-cache rebuild
	$(MAKE) -C A6-shared rebuild
	$(MAKE) -C A7-texture rebuild
