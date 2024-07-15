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

The code and model of the moment retrieval package (contents of folder `cgdetr`) are borrowed from [CGDETR](https://github.com/wjun0830/CGDETR.git);
see their respective license headers for more details: 

```txt
MIT License
CG-DETR (https://github.com/wjun0830/CGDETR)
Copyright (c) 2023 WonJun Moon

MIT License
QD-DETR (https://github.com/wjun0830/QD-DETR)
Copyright (c) 2022 WonJun Moon

MIT License
Moment-DETR (https://github.com/jayleicn/moment_detr)
Copyright (c) 2021 Jie Lei
```