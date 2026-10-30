"""Current service facade; historical entry files stay byte-for-byte stable.

The compatibility app already owns all physics, workbench and data routes.
Only public configuration discovery/preflight is adapted here. This separates
release orchestration from frozen research-source dependencies.
"""
from fastapi import APIRouter
from .api import app
from .capabilities import capabilities
from .configuration import inspect_configuration

router = APIRouter(tags=['platform'])


@router.get('/api/v1/capabilities')
def platform_capabilities():
    return capabilities()


@router.post('/api/v1/scene-check')
def check_configuration(request: dict):
    return inspect_configuration(request)


# Install before the SPA catch-all; replace only the old preflight route.
app.router.routes = [*router.routes, *[
    route for route in app.router.routes
    if getattr(route, 'path', None) not in ('/api/v1/capabilities', '/api/v1/scene-check')
]]
app.openapi_schema = None
