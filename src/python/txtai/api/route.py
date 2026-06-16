"""
Route module
"""

from fastapi.routing import APIRoute, _effective_route_context_var, get_request_handler

from .responses import ResponseFactory


class EncodingAPIRoute(APIRoute):
    """
    Extended APIRoute that encodes responses based on HTTP Accept header.
    """

    def get_route_handler(self):
        """
        Resolves a response class based on the HTTP Accept header.

        Returns:
            route handler function
        """

        # Read the effective route context NOW, while _effective_route_context_var is
        # still set by APIRoute.handle.  The ContextVar is reset in a finally-block
        # before the returned handler coroutine is ever awaited, so it must be
        # captured here (synchronously) and held in the closure below.
        effective_context = _effective_route_context_var.get()
        route = (
            effective_context
            if effective_context is not None and effective_context.original_route is self
            else self
        )

        async def handler(request):
            handler_fn = get_request_handler(
                dependant=route.dependant,
                body_field=route.body_field,
                status_code=route.status_code,
                response_class=ResponseFactory.create(request),
                response_field=getattr(route, "secure_cloned_response_field", route.response_field),
                response_model_include=route.response_model_include,
                response_model_exclude=route.response_model_exclude,
                response_model_by_alias=route.response_model_by_alias,
                response_model_exclude_unset=route.response_model_exclude_unset,
                response_model_exclude_defaults=route.response_model_exclude_defaults,
                response_model_exclude_none=route.response_model_exclude_none,
                dependency_overrides_provider=route.dependency_overrides_provider,
            )

            return await handler_fn(request)

        return handler
