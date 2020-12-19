echo killing old docker containers
docker container stop fyp_fastapi_backend
docker container rm fyp_fastapi_backend
echo building docker image
docker build -t fyp_fastapi ./fastapi_app
echo running docker container
docker run -d --name fyp_fastapi_backend -p 8080:8080 fyp_fastapi