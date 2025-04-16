docker_compose_core_path: str = 'config/docker-compose-core.yml'
with open(docker_compose_core_path, 'r') as compose_file:
    server_arch: str = compose_file.read()
with open('.env', 'r', newline='\r\n') as env_file:
    for newline in env_file:
        if newline.startswith("TOTAL_CLIENTS"):
            num_clients: int = int(newline.split('=')[1])

for i in range(num_clients):
    client_template: str = \
    f"""
  client{i}:
    container_name: client{i}
    build:
      context: .
      dockerfile: config/Dockerfile.ubuntu
    command: python client.py ${{COMMON_CLIENT_PROPERTIES}} --client-id={i}
    deploy:
      resources:
        reservations:
            devices:
              - driver: nvidia
                count: all
                capabilities: [gpu]
        limits:
          cpus: "1"
          memory: "8g"
    memswap_limit: 8G
    volumes:
      - .:/app
      - /var/run/docker.sock:/var/run/docker.sock
      - ./results:/results
    ports:
      - "{6000 + i}:{6000 + i}"
    depends_on:
      - server
    environment:
      FLASK_RUN_PORT: {6000 + i}
      container_name: client{i}
      DOCKER_HOST_IP: host.docker.internal
      NVIDIA_VISIBLE_DEVICES: all
      NVIDIA_DRIVER_CAPABILITIES: compute,utility
    stop_signal: SIGINT
    
"""
    server_arch += client_template

server_arch += "volumes:\n  grafana-storage:\n"
with open("docker-compose.yml", 'w') as compose_file:
    compose_file.write(server_arch)
