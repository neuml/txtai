"""
GGML module
"""

import ctypes
import os

import numpy as np

# Conditional import
try:
    import ggml
    from ggml import utils

    LIBGGML = True
except ImportError:
    LIBGGML = False

from ..base import ANN


class GGML(ANN):
    """
    Builds an ANN index backed by GGML.
    """

    def __init__(self, config):
        super().__init__(config)

        if not LIBGGML:
            raise ImportError('GGML is not available - install "ann" extra to enable')

    def load(self, path):
        # Create GGML Tensors
        self.backend = GGMLTensors(self.setting("gpu", True), self.setting("querysize", 64), self.setting("quantize"))

        # Load existing GGUF file
        self.backend.load(path)

    def index(self, embeddings):
        # Create GGML Tensors
        self.backend = GGMLTensors(self.setting("gpu", True), self.setting("querysize", 64), self.setting("quantize"))

        # Add embeddings data
        self.backend.index(embeddings)

        # Add id offset and index build metadata
        self.config["offset"] = embeddings.shape[0]
        self.metadata(self.settings())

    def append(self, embeddings):
        # Append embeddings to existing tensors
        self.backend.append(embeddings)

        # Update id offset and index metadata
        self.config["offset"] += embeddings.shape[0]
        self.metadata()

    def delete(self, ids):
        self.backend.delete(ids)

    def search(self, queries, limit):
        scores = self.backend.search(queries)

        # Get topn ids
        ids = np.argsort(-scores)[:, :limit]

        # Map results to [(id, score)]
        results = []
        for x, score in enumerate(scores):
            # Add results
            results.append(list(zip(ids[x].tolist(), score[ids[x]].tolist())))

        return results

    def count(self):
        return self.backend.count()

    def save(self, path):
        self.backend.save(path)

    def close(self):
        # Cleanup resources before setting backend to None
        if self.backend:
            self.backend.close()

        # Parent logic
        super().close()

    def settings(self):
        """
        Returns settings for this instance.

        Returns:
            dict
        """

        return {"ggml": ggml.__version__}


class GGMLTensors:
    """
    Interface to read and write GGML tensor data.
    """

    def __init__(self, gpu, querysize, quantize):
        """
        Creates a new GGMLTensors.

        Args:
            gpu: if GPU should be used
            querysize: query buffer size
            quantize: data quantization setting
        """

        # Settings
        self.gpu, self.querysize, self.quantize = gpu, querysize, quantize

        # GGML parameters
        self.context, self.backend = None, None
        self.buffer, self.queries, self.data, self.deletes = None, None, None, []
        self.allocator, self.graph, self.output = None, None, None

    def __del__(self):
        """
        Ensure resources are cleaned up.
        """

        # Cleanup resources
        self.close()

    def load(self, path):
        """
        Loads GGML tensors from a GGUF file at path.

        Args:
            path: path to GGUF file
        """

        # Initialize GGML objects
        self.context = self.createcontext()
        self.backend = self.createbackend()
        self.allocator = self.createallocator(self.backend)

        # Temporary context for GGUF
        context = ctypes.c_void_p()

        # Cast as ggml_context**
        params = ggml.gguf_init_params(ctx=ctypes.pointer(context), no_alloc=False)

        # Load GGUF file
        gguf = ggml.gguf_init_from_file(path.encode("utf-8"), params)

        # Load tensors from GGUF file
        self.loadtensors(context)

        # Create graph operation
        self.graph, self.output = self.creategraph()

        # Cleanup temporary resources
        ggml.gguf_free(gguf)
        ggml.ggml_free(context)

    def index(self, embeddings):
        """
        Indexes embeddings as GGML tensors.

        Args:
            embeddings: embeddings array
        """

        # Initialize GGML objects
        self.context = self.createcontext()
        self.backend = self.createbackend()
        self.allocator = self.createallocator(self.backend)

        # Create query buffer and data tensors
        self.createtensors(embeddings)

        # Create graph operation
        self.graph, self.output = self.creategraph()

    def append(self, embeddings):
        """
        Appends embeddings to GGML tensors.

        Args:
            embeddings: embeddings array
        """

        # Initialize GGML objects
        context = self.createcontext()
        backend = self.createbackend()
        allocator = self.createallocator(backend)

        # Merge embeddings tensors
        buffer, queries, data = self.mergetensors(context, backend, embeddings)
        deletes = self.deletes

        # Free existing objects
        self.close()

        # Store new objects
        self.context, self.backend, self.allocator = (context, backend, allocator)
        self.buffer, self.queries, self.data, self.deletes = (buffer, queries, data, deletes)

        # Create graph operation
        self.graph, self.output = self.creategraph()

    def delete(self, ids):
        """
        Delete ids from tensors.

        Args:
            ids: ids to delete
        """

        shape = utils.get_shape(self.data)

        # Filter any index greater than size of array
        ids = [x for x in ids if x < shape[1]]
        self.deletes.extend(ids)

    def search(self, queries):
        """
        Searches GGML tensors for the best query matches.

        Args:
            queries: queries array

        Returns:
            query results
        """

        # Process queries up to the query buffer size batches
        batches = []
        for batch in self.chunk(queries):
            # Copy queries to buffer
            ggml.ggml_backend_tensor_set(
                self.queries,
                ctypes.cast(batch.ctypes.data, ctypes.c_void_p),
                0,
                batch.nbytes,
            )

            # Run matrix multiplication operation
            ggml.ggml_backend_graph_compute(self.backend, self.graph)

            # Get size of embeddings data
            size = utils.get_shape(self.data)[1]

            # Get and return results
            results = np.zeros((batch.shape[0], size), dtype=np.float32)
            ggml.ggml_backend_tensor_get(self.output, ctypes.cast(results.ctypes.data, ctypes.c_void_p), 0, results.nbytes)

            # Clear deleted rows and add results
            results[:, self.deletes] = 0
            batches.append(results)

        # Combine batches and return as single result
        return np.concatenate(batches, axis=0)

    def count(self):
        """
        Number of elements in this GGML tensors.

        Returns:
            count
        """

        return utils.get_shape(self.data)[1] - len(self.deletes) if self.data else 0

    def save(self, path):
        """
        Saves GGML tensors as GGUF to path.

        Args:
            path: path to save
        """

        # Temporary buffer
        buffer = None

        # Init and save data tensor
        gguf = ggml.gguf_init_empty()

        # Add the data tensor
        ggml.ggml_set_name(self.data, b"data")
        ggml.gguf_add_tensor(gguf, self.data)

        # Optionally create and add the deletes tensor
        if self.deletes:
            deletes = np.array(self.deletes, dtype=np.int64)
            tensor = ggml.ggml_new_tensor_1d(self.context, ggml.GGML_TYPE_I64, deletes.shape[0])
            buffer = ggml.ggml_backend_alloc_ctx_tensors(self.context, self.backend)

            ggml.ggml_backend_tensor_set(
                tensor,
                ctypes.cast(deletes.ctypes.data, ctypes.c_void_p),
                0,
                deletes.nbytes,
            )
            ggml.ggml_set_name(tensor, b"deletes")
            ggml.gguf_add_tensor(gguf, tensor)

        # Write file and free resources
        ggml.gguf_write_to_file(gguf, path.encode("utf-8"), False)
        ggml.gguf_free(gguf)

        if buffer:
            ggml.ggml_backend_buffer_free(buffer)

    def close(self):
        """
        Closes this instance and frees resources.
        """

        if self.buffer:
            ggml.ggml_backend_buffer_free(self.buffer)
            self.buffer, self.queries, self.data, self.deletes = None, None, None, []

        if self.allocator:
            ggml.ggml_gallocr_free(self.allocator)
            self.allocator, self.graph = None, None

        if self.backend:
            ggml.ggml_backend_free(self.backend)
            self.backend = None

        if self.context:
            # Free quantization memory
            ggml.ggml_quantize_free()

            # Free context
            ggml.ggml_free(self.context)
            self.context = None

    def createcontext(self):
        """
        Creates a new GGML context.

        Returns:
            context
        """

        # Base tensor storage
        size = ggml.ggml_tensor_overhead() * 100

        # Graph storage
        size += ggml.ggml_tensor_overhead() * ggml.GGML_DEFAULT_GRAPH_SIZE + ggml.ggml_graph_overhead()

        # Create GGML context
        params = ggml.ggml_init_params(mem_size=size, no_alloc=True)
        context = ggml.ggml_init(params)

        return context

    def createbackend(self):
        """
        Creates a new GGML backend.

        Returns:
            backend
        """

        # Attempt to create an accelerated backend
        backend = ggml.ggml_backend_init_by_type(ggml.GGML_BACKEND_DEVICE_TYPE_GPU, None) if self.gpu else None

        # Fall back to CPU backend
        if not backend:
            backend = ggml.ggml_backend_cpu_init()
            ggml.ggml_backend_cpu_set_n_threads(backend, os.cpu_count())

        return backend

    def createallocator(self, backend):
        """
        Creates a new GGML allocator.

        Args:
            backend: backend device

        Returns:
            allocator
        """

        return ggml.ggml_gallocr_new(ggml.ggml_backend_get_default_buffer_type(backend))

    def createtensors(self, data):
        """
        Creates query and data tensors.

        Args:
            data: embeddings data
        """

        # Derive embeddings data tensor type
        tensortype = self.tensortype(data)

        # Queries
        self.queries = ggml.ggml_new_tensor_2d(self.context, ggml.GGML_TYPE_F32, data.shape[1], self.querysize)

        # Embeddings data
        self.data = ggml.ggml_new_tensor_2d(self.context, tensortype, data.shape[1], data.shape[0])

        # Create buffer
        self.buffer = ggml.ggml_backend_alloc_ctx_tensors(self.context, self.backend)

        # Copy embeddings data
        self.copy(data, self.data, tensortype, 0)

    def loadtensors(self, context):
        """
        Loads existing tensors from context.

        Args:
            context: ggml context
        """

        # Load data tensor
        data = ggml.ggml_get_tensor(context, b"data")
        if data:
            # Queries
            shape = utils.get_shape(data)
            self.queries = ggml.ggml_new_tensor_2d(self.context, ggml.GGML_TYPE_F32, shape[0], self.querysize)

            # Embeddings data
            self.data = ggml.ggml_dup_tensor(self.context, data)

            # Create buffer
            self.buffer = ggml.ggml_backend_alloc_ctx_tensors(self.context, self.backend)

            # Copy tensor data to backend
            ggml.ggml_backend_tensor_set(self.data, ggml.ggml_get_data(data), 0, ggml.ggml_nbytes(data))

        # Load deletes tensor
        data = ggml.ggml_get_tensor(context, b"deletes")
        if data:
            # Convert to a NumPy array
            shape = utils.get_shape(data)
            deletes = np.ctypeslib.as_array(ctypes.cast(ggml.ggml_get_data(data), ctypes.POINTER(ctypes.c_int64)), (shape[0],))
            self.deletes = deletes.tolist()

    def mergetensors(self, context, backend, data):
        """
        Merges new embeddings data.

        Args:
            context: new context
            backend: new backend
            data: embeddings data

        Returns:
            buffer, queries, data
        """

        # Derive embeddings data tensor type
        tensortype = self.tensortype(data)

        # Queries
        queries = ggml.ggml_new_tensor_2d(context, ggml.GGML_TYPE_F32, data.shape[1], self.querysize)

        # Embeddings data with space for both existing and new data
        shape = utils.get_shape(self.data)
        merge = ggml.ggml_new_tensor_2d(context, tensortype, data.shape[1], data.shape[0] + shape[1])

        # Create new buffer
        buffer = ggml.ggml_backend_alloc_ctx_tensors(context, backend)

        # Copy existing embeddings data
        self.copy(self.data, merge, tensortype, 0)

        # Copy new embeddings data
        self.copy(data, merge, tensortype, ggml.ggml_nbytes(self.data))

        return buffer, queries, merge

    def creategraph(self):
        """
        Creates a new GGML graph.

        Returns:
            graph
        """

        # Create matrix multiply graph operation
        graph = ggml.ggml_new_graph(self.context)

        # Graph operation
        output = ggml.ggml_mul_mat(self.context, self.data, self.queries)

        # Setup and allocate graph storage
        ggml.ggml_build_forward_expand(graph, output)
        ggml.ggml_gallocr_alloc_graph(self.allocator, graph)

        return graph, output

    def tensortype(self, data):
        """
        Gets the best matching tensor type for input data.

        Args:
            data: embeddings data

        Returns:
            best matching GGML data type
        """

        # Read tensor type
        tensortype = self.quantize
        tensortype = "Q8_0" if isinstance(tensortype, bool) else f"Q{int(tensortype)}_0" if isinstance(tensortype, int) else tensortype
        tensortype = tensortype.upper() if tensortype else "F32"

        # Validate tensor type
        if not hasattr(ggml, f"GGML_TYPE_{tensortype}"):
            raise ValueError(f"Invalid tensor type {tensortype}")

        # Get tensor type
        tensortype = getattr(ggml, f"GGML_TYPE_{tensortype}")

        # Validate quantization block size
        blocksize = ggml.ggml_blck_size(tensortype)
        if data.shape[1] % blocksize != 0:
            raise ValueError(
                f'Invalid quantization configuration "{self.quantize}" with {data.shape[1]} dimensions. Must be a multiple of {blocksize}.'
            )

        return tensortype

    def copy(self, inputs, outputs, tensortype, offset):
        """
        Copies input data to backend. Quantizes to desired tensor type, if necessary.

        Args:
            inputs: input tensor
            outputs: output tensor
            tensortype: desired tensor type
            offset: data offset index for storage into outputs
        """

        if not isinstance(inputs, np.ndarray):
            # GGML tensor
            work, size = ggml.ggml_get_data(inputs), ggml.ggml_nbytes(inputs)
        elif tensortype == ggml.GGML_TYPE_F32:
            # No quantization needed
            work, size = inputs.ctypes.data, inputs.nbytes
        else:
            # Work array will be garbage collected by Python
            work = (ctypes.c_float * inputs.shape[0] * inputs.shape[1])()

            # Quantize vector data
            size = ggml.ggml_quantize_chunk(
                tensortype, ctypes.cast(inputs.ctypes.data, ctypes.POINTER(ctypes.c_float)), work, 0, inputs.shape[0], inputs.shape[1], None
            )

        # Copy data to tensor
        ggml.ggml_backend_tensor_set(outputs, work, offset, size)

    def chunk(self, queries):
        """
        Splits quries into separate batch sizes specified by size.

        Args:
            queries: queries

        Returns:
            list of evenly sized batches with the last batch having the remaining elements
        """

        return [queries[x : x + self.querysize] for x in range(0, len(queries), self.querysize)]
