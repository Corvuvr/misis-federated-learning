## Требования
- ОС: Linux / Windows with WSL2 enabled
- Установленныe [драйвер nvidia](https://www.nvidia.com/en-us/drivers/), Docker и [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
## Запуск
В linux shell:  
1. Запустить Docker
2. Загрузить проект и, находясь в папке проекта, выполнить:
```shell
docker compose build
docker compose up
```
## Пользование
После запуска открыть в браузере: `localhost:3000` - на странице появится интерфейс Graphana с информацией для мониторинга. 
