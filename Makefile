# Weighted KT coding benchmarks. C89, requires only libm.
CC      = cc
CFLAGS  = -std=c89 -pedantic -Wall -Wextra -O2
LDLIBS  = -lm

all: wkt_bench wkt_records

wkt_bench: src/harness.c src/wkt.c src/wkt.h
	$(CC) $(CFLAGS) -Isrc -o $@ src/harness.c src/wkt.c $(LDLIBS)

wkt_records: src/records.c src/wkt.c src/wkt.h
	$(CC) $(CFLAGS) -Isrc -o $@ src/records.c src/wkt.c $(LDLIBS)

clean:
	rm -f wkt_bench wkt_records

.PHONY: all clean
