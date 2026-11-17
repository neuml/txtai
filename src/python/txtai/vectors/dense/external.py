"""
External module
"""

import functools
import hashlib
import inspect
import marshal
import os
import re
import types
import uuid

from ...util import Library, Resolver

from ..base import Vectors

# Core library imports
np = Library().numpy()


def _fingerprint(value, seen=frozenset(), depth=0):
    """
    Builds a stable token identifying a value, or None when the value cannot be identified
    deterministically (for example, it closes over an object whose repr contains a memory
    address). Tokens are stable across runs for the same value, so the same transform
    produces the same token while a changed transform produces a different one.

    The token covers the transform's code and its plain-data state (numbers, strings,
    collections, arrays). Values that cannot be identified, such as model objects or
    API clients referenced by the transform, are left out of the token rather than
    failing it, so those transforms keep working with checkpoints. That also means a
    change confined to such an object alone is not detected, the same limitation that
    already applies to model paths in the vectors configuration.

    Args:
        value: value to identify
        seen: ids of values already being fingerprinted, guards against reference cycles
        depth: current recursion depth, guards against deeply nested values

    Returns:
        stable token for the value, or None when it cannot be identified
    """

    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        return repr(value)

    if isinstance(value, np.ndarray):
        digest = hashlib.sha256(value.tobytes()).hexdigest()
        return f"ndarray({value.dtype},{value.shape},{digest})"

    if isinstance(value, types.ModuleType):
        return f"module:{value.__name__}"

    if isinstance(value, (list, tuple, dict, set, frozenset)):
        return _fingerprintcollection(value, seen, depth)

    if isinstance(value, functools.partial):
        return _partialfingerprint(value, seen, depth)

    if isinstance(value, types.FunctionType):
        return _functionfingerprint(value, seen, depth)

    if isinstance(value, types.MethodType):
        return _methodfingerprint(value, seen, depth)

    if callable(value):
        return _callablefingerprint(value, seen, depth)

    return _reprfingerprint(value)


def _fingerprintcollection(value, seen, depth):
    """
    Builds a stable token for a collection value.

    Args:
        value: collection to identify
        seen: ids of values already being fingerprinted
        depth: current recursion depth

    Returns:
        stable token for the collection, or None when it cannot be identified
    """

    if depth > 10 or id(value) in seen:
        return None

    seen = seen | {id(value)}

    if isinstance(value, dict):
        return _fingerprintdict(value, seen, depth)

    if isinstance(value, (set, frozenset)):
        return _fingerprintset(value, seen, depth)

    return _fingerprintsequence(value, seen, depth)


def _fingerprintsequence(value, seen, depth):
    """
    Builds a stable token for a list or tuple value.

    Args:
        value: sequence to identify
        seen: ids of values already being fingerprinted
        depth: current recursion depth

    Returns:
        stable token for the sequence, or None when it cannot be identified
    """

    parts = [_fingerprint(item, seen, depth + 1) for item in value]

    parts = [part for part in parts if part is not None]

    return f"{type(value).__name__}({','.join(parts)})"


def _fingerprintdict(value, seen, depth):
    """
    Builds a stable token for a dict value, with keys in a deterministic order. Entries
    whose values cannot be identified are left out, as documented in _fingerprint.

    Args:
        value: dict to identify
        seen: ids of values already being fingerprinted
        depth: current recursion depth

    Returns:
        stable token for the dict, or None when it cannot be identified
    """

    parts = []
    for key in sorted(value, key=repr):
        keytoken = _fingerprint(key, seen, depth + 1)
        valuetoken = _fingerprint(value[key], seen, depth + 1)
        if keytoken is None or valuetoken is None:
            continue
        parts.append(f"{keytoken}:{valuetoken}")

    return "{" + ",".join(parts) + "}"


def _fingerprintset(value, seen, depth):
    """
    Builds a stable token for a set or frozenset value, with members in a deterministic order.

    Args:
        value: set to identify
        seen: ids of values already being fingerprinted
        depth: current recursion depth

    Returns:
        stable token for the set, or None when it cannot be identified
    """

    parts = [_fingerprint(item, seen, depth + 1) for item in value]

    parts = [part for part in parts if part is not None]

    return f"{type(value).__name__}({','.join(sorted(parts))})"


def _partialfingerprint(value, seen, depth):
    """
    Builds a stable token for a functools.partial value. Arguments that cannot be
    identified are left out, as documented in _fingerprint.

    Args:
        value: partial to identify
        seen: ids of values already being fingerprinted
        depth: current recursion depth

    Returns:
        stable token for the partial, or None when it cannot be identified
    """

    if depth > 10:
        return None

    func = _fingerprint(value.func, seen, depth + 1)

    if func is None:
        return None

    parts = [func]
    for arg in (value.args, value.keywords):
        token = _fingerprint(arg, seen, depth + 1)
        if token is not None:
            parts.append(token)

    return f"partial({','.join(parts)})"


def _functionfingerprint(func, seen=frozenset(), depth=0):
    """
    Builds a stable token for a function, covering its code, defaults, closure values and
    referenced globals. A function whose closed-over state changes (for example, a flag
    that switches its behavior) produces a different token.

    Args:
        func: function to identify
        seen: ids of functions already being fingerprinted
        depth: current recursion depth

    Returns:
        stable token for the function, or None when it cannot be identified
    """

    if depth > 10:
        return None

    if id(func) in seen:
        return f"func:{func.__module__}.{func.__qualname__}"

    seen = seen | {id(func)}

    try:
        code = hashlib.sha256(marshal.dumps(func.__code__)).hexdigest()
    except Exception:  # pylint: disable=broad-exception-caught
        return None

    parts = [func.__module__, func.__qualname__, code]

    for defaults in (func.__defaults__, func.__kwdefaults__):
        token = _fingerprint(defaults, seen, depth + 1)
        if token is not None:
            parts.append(token)

    try:
        closure = inspect.getclosurevars(func)
    except Exception:  # pylint: disable=broad-exception-caught
        return None

    for name in sorted(closure.nonlocals):
        token = _fingerprint(closure.nonlocals[name], seen, depth + 1)
        if token is not None:
            parts.append(f"{name}={token}")

    for name in sorted(closure.globals):
        token = _fingerprint(closure.globals[name], seen, depth + 1)
        if token is not None:
            parts.append(f"{name}={token}")

    return "func(" + "|".join(parts) + ")"


def _methodfingerprint(value, seen, depth):
    """
    Builds a stable token for a bound method, from its function code and its instance
    when the instance can be identified. An instance that cannot be identified is left
    out, as documented in _fingerprint.

    Args:
        value: bound method to identify
        seen: ids of values already being fingerprinted
        depth: current recursion depth

    Returns:
        stable token for the bound method, or None when it cannot be identified
    """

    if depth > 10:
        return None

    func = _functionfingerprint(value.__func__, seen, depth)

    if func is None:
        return None

    parts = [func]
    obj = _fingerprint(value.__self__, seen, depth + 1)
    if obj is not None:
        parts.append(obj)

    return f"method({','.join(parts)})"


def _callablefingerprint(value, seen, depth):
    """
    Builds a stable token for a callable instance, from its __call__ code and its state
    when the state can be identified. State that cannot be identified is left out, as
    documented in _fingerprint.

    Args:
        value: callable instance to identify
        seen: ids of values already being fingerprinted
        depth: current recursion depth

    Returns:
        stable token for the callable instance, or None when it cannot be identified
    """

    if depth > 10:
        return None

    call = value.__call__
    if not isinstance(call, types.MethodType):
        return None

    func = _functionfingerprint(call.__func__, seen, depth)

    if func is None:
        return None

    parts = [f"{type(value).__module__}.{type(value).__qualname__}", func]
    state = _fingerprint(vars(value), seen, depth + 1) if hasattr(value, "__dict__") else None
    if state is not None:
        parts.append(state)

    return f"callable({','.join(parts)})"


def _reprfingerprint(value):
    """
    Builds a stable token from a value's repr, rejecting reprs that contain a memory
    address, since those change from run to run and would be mistaken for a stable
    identity.

    Args:
        value: value to identify

    Returns:
        stable token for the value, or None when its repr is not stable
    """

    text = repr(value)

    if re.search(r"0x[0-9a-fA-F]+", text):
        return None

    return f"repr:{text}"


def _transformfingerprint(transform):
    """
    Builds a stable fingerprint for an external transform callable, or None when the
    callable cannot be identified deterministically.

    Args:
        transform: transform callable to identify

    Returns:
        fingerprint for the transform, or None
    """

    token = _fingerprint(transform)
    return hashlib.sha256(token.encode("utf-8")).hexdigest() if token is not None else None


class External(Vectors):
    """
    Builds vectors using an external method. This can be a local function or an external API call.
    """

    def __init__(self, config, scoring, models):
        super().__init__(config, scoring, models)

        # Lookup and resolve transform function
        self.transform = self.resolve(config.get("transform"))

    def loadmodel(self, path):
        return None

    def encode(self, data, category=None):
        # Call external transform function, if available and data not already an array
        # Batching is handed by the external transform function
        if self.transform and data and not isinstance(data[0], np.ndarray):
            data = self.transform(data)

        # Cast to float32
        return data.astype(np.float32) if isinstance(data, np.ndarray) else np.array(data, dtype=np.float32)

    def transformid(self):
        """
        Returns a fingerprint identifying the external transform function. The transform
        determines the produced vectors but is not part of the configuration, so it is
        included in the vectors uid to keep checkpoint recovery from reusing vectors
        produced by a different transform.

        A transform that cannot be identified gets a unique value on every call, so no
        checkpoint is ever trusted for it and its vectors are always recomputed.

        Returns:
            transform fingerprint, or None when there is no transform
        """

        if not self.transform:
            return None

        fingerprint = _transformfingerprint(self.transform)

        if fingerprint is not None:
            return fingerprint

        # Cannot identify the transform - fail closed so no checkpoint is ever trusted for it
        return f"unknown:{uuid.uuid4()}"

    def resolve(self, transform):
        """
        Resolves a transform function.

        Args:
            transform: transform function

        Returns:
            resolved transform function
        """

        if transform:
            # Check if transform function resolution is allowed
            if isinstance(transform, str) and os.environ.get("ALLOW_RESOLVE_TRANSFORM", "False") not in ("True", "1"):
                raise ImportError(
                    (
                        "External transform function resolution is disabled. "
                        "Set the env variable `ALLOW_RESOLVE_TRANSFORM=True` to enable transform function resolution. "
                        "This should only be done for trusted and/or reviewed code. "
                    )
                )

            # Resolve transform instance, if necessary
            transform = Resolver()(transform) if isinstance(transform, str) else transform

            # Get function or callable instance
            transform = transform if isinstance(transform, types.FunctionType) else transform()

        return transform
