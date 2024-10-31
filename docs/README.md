# Building Docs

```sh
pip install -r requirements.txt
pip install livereload pyppeteer
```

To build the documentation and serve it to `localhost:8010`:

```sh
python serve.py -p 8010
```

To build the documentation as a pdf:

```sh
./buildpdf
```