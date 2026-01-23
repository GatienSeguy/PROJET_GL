import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "main_test:app",
        host="127.0.0.1",
        port=8001,
        reload=False,   
        workers=1       
    )
