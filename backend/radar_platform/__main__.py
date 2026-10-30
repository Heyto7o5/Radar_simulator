import os

import uvicorn


if __name__ == "__main__":
    uvicorn.run(
        "radar_platform.api:app",
        host=os.environ.get("RADAR_API_HOST", "0.0.0.0"),
        port=int(os.environ.get("RADAR_API_PORT", "8000")),
        reload=os.environ.get("RADAR_RELOAD", "0") == "1",
    )
