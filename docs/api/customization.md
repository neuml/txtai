# Customization

The txtai API has a number of features out of the box that are designed to help get started quickly. API services can also be augmented with custom code and functionality. The two main ways to do this are with extensions and dependencies.

Extensions add a custom endpoint. Dependencies add middleware that executes with each request. See the sections below for more.

## Extensions

While the API is extremely flexible and complex logic can be executed through YAML-driven workflows, some may prefer to create an endpoint in Python. API extensions define custom Python endpoints that interact with txtai applications. 

See the link below for a detailed example.

| Notebook  | Description  |       |
|:----------|:-------------|------:|
| [Custom API Endpoints](https://github.com/neuml/txtai/blob/master/examples/51_Custom_API_Endpoints.ipynb) | Extend the API with custom endpoints | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuml/txtai/blob/master/examples/51_Custom_API_Endpoints.ipynb) |

## Dependencies

txtai has a default API token authorization method that works well in many cases. Dependencies can also add custom logic with each request. This could be an additional authorization step and/or an authentication method. 

See the link below for a detailed example.

| Notebook  | Description  |       |
|:----------|:-------------|------:|
| [API Authorization and Authentication](https://github.com/neuml/txtai/blob/master/examples/54_API_Authorization_and_Authentication.ipynb) | Add authorization, authentication and middleware dependencies to the API | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuml/txtai/blob/master/examples/54_API_Authorization_and_Authentication.ipynb) |
