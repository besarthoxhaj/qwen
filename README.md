# Qwen Chat


```sh
$ mkdir -p ~/miniconda3
$ wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
$ bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
$ rm ~/miniconda3/miniconda.sh
$ source ~/miniconda3/bin/activate
$ conda init --all
```


```sh
$ conda create --name sns python=3.12.4 -y
$ conda activate sns
$ pip install fastapi torch transformers uvicorn pydantic
$ python server.py
```


```sh

```