.PHONY: bootstrap up

all: bootstrap up

bootstrap:
	python3 -m pip -q install -U "huggingface_hub[cli]"
	@[ -f .env ] || cp .env.example .env
	chmod +x ./get_model.sh
	./get_model.sh

up:
	chmod +x ./run_toddric.sh
	./run_toddric.sh

