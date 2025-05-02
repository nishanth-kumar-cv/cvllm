.PHONY: preprocess train serve docker-build deploy clean

preprocess:
	./scripts/run_preprocessing.sh

train:
	./scripts/train.sh

serve:
	./scripts/serve.sh

docker-build:
	docker build -t mistral-finetune .

deploy:
	cd terraform && terraform init && terraform apply -auto-approve

clean:
	cargo clean
	rm -rf __pycache__ python_finetune/__pycache__ .terraform
