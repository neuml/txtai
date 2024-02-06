"""
Route module
"""

from fastapi.routing import APIRoute, get_request_handler

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

        async def handler(request):
            route = get_request_handler(
                dependant=self.dependant,
                body_field=self.body_field,
                status_code=self.status_code,
                response_class=ResponseFactory.create(request),
                response_field=self.secure_cloned_response_field,
                response_model_include=self.response_model_include,
                response_model_exclude=self.response_model_exclude,
                response_model_by_alias=self.response_model_by_alias,
                response_model_exclude_unset=self.response_model_exclude_unset,
                response_model_exclude_defaults=self.response_model_exclude_defaults,
                response_model_exclude_none=self.response_model_exclude_none,
                dependency_overrides_provider=self.dependency_overrides_provider,
            )

            return await route(request)

        return handler
