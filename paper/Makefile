# Build the TSAM NeurIPS paper draft.
#
# Default target: pdf
#
# Requirements:
#   - latexmk (recommended) OR pdflatex + bibtex
#   - The neurips_2024.sty file ships in this directory
#
# Usage:
#   make            # build paper.pdf via latexmk
#   make clean      # remove aux files, keep paper.pdf
#   make distclean  # remove everything including paper.pdf

PAPER := main
BIB   := references

.PHONY: pdf clean distclean

pdf:
	@if command -v latexmk >/dev/null 2>&1; then \
		latexmk -pdf -bibtex -interaction=nonstopmode -halt-on-error $(PAPER).tex; \
	else \
		echo ">> latexmk not found; falling back to pdflatex+bibtex"; \
		pdflatex -interaction=nonstopmode -halt-on-error $(PAPER).tex; \
		bibtex $(PAPER); \
		pdflatex -interaction=nonstopmode -halt-on-error $(PAPER).tex; \
		pdflatex -interaction=nonstopmode -halt-on-error $(PAPER).tex; \
	fi

clean:
	@rm -f *.aux *.log *.out *.bbl *.blg *.fls *.fdb_latexmk *.synctex.gz *.toc

distclean: clean
	@rm -f $(PAPER).pdf
