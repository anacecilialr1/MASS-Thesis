all: main count spell check

main: FORCE | build
	TEXINPUTS=.: \
	BIBINPUTS=.: \
	max_print_line=1048576 \
	latexmk \
		-pdf \
		--output-directory=build \
		--interaction=batchmode \
		--halt-on-error \
		main.tex
		
#; cp build/main.pdf Main.pdf

count:
	texcount chapter/*.tex *.tex

spell:
	cat chapter/*.tex *.tex | \
	aspell list --mode=tex --master=en_US | \
	aspell list --mode=tex --master=es | sort | uniq

build:
	mkdir -p build

clean:
	rm -rf build
	rm -rf $(shell find /private/var/folders -name "par-*" -type d 2>/dev/null)
# cleans cached files from BIBER?

check:
	which biber
	biber --version
	which latexmk
	latexmk --version

FORCE:

.PHONY: all clean