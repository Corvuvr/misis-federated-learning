## Требования
- OC: Linux / Windows
- Установлен Docker
## Требования для работы с GPU (дополнительно)
- ОС: Linux / Windows с включённым WSL2
- Установленныe [драйвер nvidia](https://www.nvidia.com/en-us/drivers/), и [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
## Запуск (федеративно)
1. Загрузить проект
2. Запустить Docker
3. Редактировать `.env`-файл интересующим образом.
4. Сгенерировать файл `docker-compose.yml` командой:
```shell
python helpers/generate_docker_compose.py
```
5. Находясь в папке проекта, выполнить из оболочки:
```shell
docker compose build
docker compose up
```
После запуска открыть в браузере: `localhost:3000` - на странице появится интерфейс Graphana с информацией для мониторинга. 

## Запуск (локально)
Скачать [пример](examples/client-solo.ipynb) на Google Drive и открыть его в Colab, или повторить шаги из примера вручную.