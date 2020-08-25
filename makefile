SPACYMODEL 	:= en_core_web_lg

bootstrap: setupEnvironment

install:
	python -m spacy download ${SPACYMODEL}
	pip install -r requirements.dev.txt

setupEnvironment:
	python3.8 -m venv venv
	source venv/bin/activate	
	pip install pip-tools

updateDependencies:
	pip-compile requirements.dev.in > requirements.dev.txt
	