from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from starlette.middleware.wsgi import WSGIMiddleware
from dash import Dash, dcc, html
from dash import dash
from dash.dependencies import Input, Output
from starlette.applications import Starlette
from starlette.types import ASGIApp, Receive, Scope, Send
import uvicorn

app_fastapi = FastAPI()
templates = Jinja2Templates(directory="templates")

@app_fastapi.get("/api")
def api_endpoint():
    return {"message": "Hello from FastAPI!"}

app_dash = Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=["https://codepen.io/chriddyp/pen/bWLwgP.css"])

@app_dash.callback(
   Output('my-div', 'children'),
   [Input('my-input', 'value')]
)
def update_output(value):
    return 'You have entered "{}"'.format(value)

app_dash.layout = html.Div([
    dcc.Input(id='my-input', value='Dash App', type='text'),
    html.Div(id='my-div')
])

async def dash_asgi(request: Request, receive: Receive, send: Send) -> None:
    await app_dash.server(request.scope)(receive, send)

@app_fastapi.get("/dashboard", response_class=HTMLResponse)
async def dash_endpoint(request: Request):
    content = app_dash.index()
    return HTMLResponse(content=content)

if __name__ == "__main__":
    from hypercorn.config import Config

    config = Config()
    config.bind = ["127.0.0.1:8000"]
    uvicorn.run(app_fastapi, config=config, log_level="info")
