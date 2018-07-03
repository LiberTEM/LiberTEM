all: foo.d.ts

foo.json: source.py
	python source.py | python -m json.tool > foo.json

foo.d.ts: foo.json
	./node_modules/.bin/json2ts -i foo.json -o foo.d.ts
