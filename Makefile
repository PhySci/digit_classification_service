build:
	docker build -f docker/Dockerfile -t ml-service:latest src

run:
	docker run --rm -it -p 8000:8000 ml-service:latest