# slider_server.py
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

app = FastAPI()

# Allow API calls from local clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

slider_value = {"value": 90}  # default

class ValueModel(BaseModel):
    value: int

@app.get("/")
def index():
    """Simple HTML page with a JavaScript slider UI"""
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Servo Slider (0–180)</title>
        <style>
            body {{
                font-family: sans-serif;
                text-align: center;
                margin-top: 100px;
            }}
            input[type=range] {{
                width: 80%;
                height: 20px;
            }}
            #val {{
                font-size: 24px;
                margin-top: 10px;
            }}
        </style>
    </head>
    <body>
        <h1>Servo Slider (0–180)</h1>
        <input type="range" id="slider" min="0" max="180" value="{slider_value['value']}">
        <div id="val">{slider_value['value']}</div>

        <script>
            const slider = document.getElementById("slider");
            const valDisplay = document.getElementById("val");

            // update local display and send value to server
            slider.addEventListener("input", async () => {{
                valDisplay.textContent = slider.value;
                await fetch("/set_value", {{
                    method: "POST",
                    headers: {{ "Content-Type": "application/json" }},
                    body: JSON.stringify({{ value: parseInt(slider.value) }})
                }});
            }});

            // periodic refresh (in case client changes it)
            async function refreshValue() {{
                const res = await fetch("/get_value");
                const data = await res.json();
                if (slider.value != data.value) {{
                    slider.value = data.value;
                    valDisplay.textContent = data.value;
                }}
            }}
            setInterval(refreshValue, 1000);
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html)


@app.get("/get_value")
def get_value():
    return slider_value

@app.post("/set_value")
def set_value(v: ValueModel):
    slider_value["value"] = max(0, min(180, v.value))  # clamp range
    return slider_value


if __name__ == "__main__":
    print("Web slider server running at http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)
