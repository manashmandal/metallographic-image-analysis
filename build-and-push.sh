echo $DOCKER_PAT | docker login -u jackspicer --password-stdin
docker build -t jackspicer/metallographic-image-analysis:latest .
docker push jackspicer/metallographic-image-analysis:latest
