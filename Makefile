.PHONY: bootstrap up

bootstrap:
	python3 -m pip -q install -U "huggingface_hub[cli]"
	@[ -f .env ] || cp .env.example .env
	./get_model.sh

up:
	./run_toddric.sh

