# swissADT


## Installation
Check out the repository:
```
git clone https://github.com/fischerl92/swissADT.git
cd swissADT
pip install -e .
```

## Run Demo

```
OPENAI_API_KEY=<your_key> streamlit run app.py
```


## Docker

Build the docker image:

```
docker build -t swiss-adt .
```

Run demo:

```
OPENAI_API_KEY=<your_key> docker compose up
```

## ☑️ LICENSE
The moment retrieval pipeline is borrowed from [CGDETR](https://github.com/wjun0830/CGDETR.git). We reorganized their code in the `cgdetr` package to make it installable.