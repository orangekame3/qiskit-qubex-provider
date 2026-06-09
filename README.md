# qiskit-qubex-provider

Qiskit provider for Qubex targets and local Qiskit execution.

```python
from qiskit import QuantumCircuit, transpile
from qiskit_qubex_provider import QubexProvider

provider = QubexProvider(num_qubits=2, coupling_map=[(0, 1)])
backend = provider.get_backend()

circuit = QuantumCircuit(2, 2)
circuit.h(0)
circuit.cx(0, 1)
circuit.measure([0, 1], [0, 1])

transpiled = transpile(circuit, backend)
job = backend.run(transpiled, shots=1024)
print(job.result().get_counts())
```

You can also pass a Qubex-like system object with `qubits` and `cr_targets`
attributes. The provider converts that metadata into a Qiskit `Target` so
circuits can be transpiled against Qubex topology and qubit properties.

```python
provider = QubexProvider(qubex_experiment_system)
backend = provider.get_backend()
sampler = provider.get_sampler()
estimator = provider.get_estimator()
```

`qubex` is intentionally not a hard package dependency because the Qubex
repository is commonly installed from a local checkout:

```bash
uv pip install -e ../qubex
```

Some Qubex imports currently need optional local backend packages as well, for
example:

```bash
uv pip install -e ../qubex/packages/qxdriver-quel1
```

For hardware execution, pass an executor object with a `run(circuits,
**options)` method through `QubexProvider(..., executor=...)`. Without an
executor, `backend.run(...)` uses Qiskit's local `BasicSimulator`.
