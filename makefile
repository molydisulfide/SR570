test: *.py
	grep -r BUG .
	python3 -m pytest test_controller.py

callgraph.png: *.py
	pyan3 *.py -c --uses --no-defines --grouped --dot | dot -Tpng > callgraph.png
