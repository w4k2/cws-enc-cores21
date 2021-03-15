work:
	./analyze.py
	python -W ignore reference.py
	./post.py

clean:
	rm results/*

paper:
	pdflatex report.tex

exp:
	python -W ignore reference.py
	./post.py
	pdflatex report.tex

post:
	./post.py
	pdflatex report.tex
