build:
	docker build -f docker/Dockerfile -t ml-service:latest .

run:
	docker run --rm -p 8050:8050 ml-service:latest

stop:
	docker stop ml-serivce:latest