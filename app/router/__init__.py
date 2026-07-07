"""Router package exports.

Expose classifier and model_selector so other modules can import
`app.router.classifier` and `app.router.model_selector` via the package.
"""

from . import classifier, model_selector

__all__ = ["classifier", "model_selector"]
