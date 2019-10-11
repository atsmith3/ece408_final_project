TARGET=report

$(TARGET).pdf: $(TARGET).tex Makefile
	pdflatex $(TARGET).tex

clean::
	rm -f *.aux *.blg *.bbl *.log *.tmp *.xref *.idv *.4og *.4ct *.lg *.4tc *~ *.out

clear: clean
	rm -f $(TARGET).pdf
