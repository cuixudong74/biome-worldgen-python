import uvicorn

from biome_worldgen import create_app

app = create_app()


@app.get("/")
def root():
    return {"service": "biome-worldgen", "docs": "/docs"}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
