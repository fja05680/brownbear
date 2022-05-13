# from brownbear/docs

# remove old html
rm html -fr

# generate html
pdoc --html ../../brownbear/

# generate markdown extra
pdoc --pdf ../../brownbear/ > brownbear.txt

# generate pdf from markdown extra
pandoc --pdf-engine=xelatex brownbear.txt -o brownbear.pdf

echo Done.
