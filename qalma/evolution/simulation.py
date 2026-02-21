"""
This module defines the `Simulation` dataclass containing the
state and the result of a simulation.
`Simulation` objects can be serialized both as Python pickle and
as  HDF5 files.

To serialize a Simulation object, use the method `save_hdf5`:

```
sim.save_hdf5(filename)
```

To load back the simulation, use the classmethod `load_hdf5`:

```
sim = Simulation.load_hdf5(filename)
```

"""

import logging
import pickle
from dataclasses import dataclass
from typing import Any, Dict, List

import h5py
import numpy as np

from qalma.operators.basic import Operator


def store_hdf5_dict(group: h5py.Group, data_dict: Dict[str, Any]):
    """
    Store data in a Python dict in a group of a hdf5 file.

    Basic types and numpy.ndarray values in data_dict are stored as
    native HDF5 data types. Other objects are stored as pickle
    byte streams.

    """
    for key, value in data_dict.items():
        key_str = str(key)
        if isinstance(value, str):
            dset = group.create_dataset(
                key_str, data=value, dtype=h5py.string_dtype(encoding="utf-8")
            )
            dset.attrs["encoding"] = "utf-8"
        elif any(isinstance(value, t) for t in [int, float, bool, np.ndarray]):
            dset = group.create_dataset(key_str, data=value)
        elif isinstance(value, (list, tuple)) and all(
            isinstance(
                x,
                (
                    int,
                    float,
                    bool,
                    str,
                ),
            )
            for x in value
        ):
            group.create_dataset(key_str, data=np.array(value))
        else:
            data = np.frombuffer(pickle.dumps(value), dtype=np.uint8)
            dset = group.create_dataset(
                key_str, shape=(1,), dtype=h5py.vlen_dtype(np.dtype("V1"))
            )
            dset.attrs["pickled"] = True
            dset[0] = data


def load_hdf5_dict(group: h5py.Group) -> Dict[str, Any]:
    """
    Load data from a Python dict stored as a group in a hdf5 file.
    """
    loaded_dict = {}
    for key_str in group.keys():
        dataset = group[key_str]
        raw_data = dataset[()]
        if dataset.dtype.kind == "O":
            if isinstance(raw_data, (bytes, np.bytes_)):
                if "encoding" in dataset.attrs:
                    loaded_dict[key_str] = raw_data.decode("utf-8")
                    continue
        if dataset.attrs.get("pickled", False):
            try:
                loaded_dict[key_str] = pickle.loads(dataset[0].tobytes())
                continue
            except (pickle.UnpicklingError, EOFError, ValueError):
                logging.warning(
                    " key %s was marked as containing a pickled object, but cannot be loaded.",
                    key_str,
                )
                pass
        loaded_dict[key_str] = raw_data
        continue

    return loaded_dict


@dataclass
class Simulation:
    """
    Hold the state and result of a simulation.
    """

    parameters: Dict[Any, Any]
    stats: Dict[Any, Any]
    time_span: List[float]
    expect_ops: Dict[Any, Any]
    states: List[Operator]

    def save_hdf5(self, filename: str):
        """
        Serialize the object as an hdf5 file
        """
        try:
            with h5py.File(filename, "w") as f:
                store_hdf5_dict(f.create_group("parameters"), self.parameters)
                store_hdf5_dict(f.create_group("stats"), self.stats)
                store_hdf5_dict(f.create_group("expect obs"), self.expect_ops)
                time_span = self.time_span
                f.create_dataset("time span", data=np.array(time_span))
                states = self.states
                if states:
                    states_group = f.create_group("states")
                    system = states[0].system
                    if system is not None:
                        data = np.frombuffer(pickle.dumps(system), dtype=np.uint8)
                        dset = f.create_dataset(
                            "system", shape=(1,), dtype=h5py.vlen_dtype(np.dtype("V1"))
                        )
                        dset[0] = data

                        def serialize_state(rho, t):
                            rho._set_system_(None)
                            data = pickle.dumps(
                                (
                                    rho,
                                    t,
                                )
                            )
                            rho._set_system_(system)
                            return data

                    else:

                        def serialize_state(rho, t):
                            return pickle.dumps((rho, t))

                    for i, rho in enumerate(states):
                        states_group.create_dataset(
                            "rho_" + f"{i}".rjust(6, "0"),
                            shape=(1,),
                            compression="gzip",
                            dtype=h5py.vlen_dtype(np.dtype("V1")),
                        )[0] = np.frombuffer(
                            serialize_state(rho, time_span[i]), dtype=np.uint8
                        )
        except (PermissionError,) as exc:
            logging.warning(
                "The object could not be stored in %s. (%s)", filename, str(exc)
            )
            return

    @classmethod
    def load_hdf5(cls, filename: str):
        """
        Load an object serialized as an hdf5 file
        """
        try:
            with h5py.File(filename, "r") as f:
                parameters = load_hdf5_dict(f["parameters"])
                stats = load_hdf5_dict(f["stats"])
                expect_ops = load_hdf5_dict(f["expect obs"])
                stored_time_span = f["time span"]
                # Ensure that time span follows the right order
                for key, data in expect_ops.items():
                    expect_ops[key] = np.array(
                        [
                            q
                            for t, q in sorted(
                                zip(stored_time_span, data), key=lambda x: x[0]
                            )
                        ]
                    )

                time_span = sorted(stored_time_span)

                try:
                    system_bytes = f["system"][0]
                    system = pickle.loads(
                        system_bytes.tobytes()
                        if isinstance(system_bytes, np.void)
                        else system_bytes
                    )
                except KeyError:
                    system = None

                states_group = f.get("states", None)
                states_and_times = []
                if states_group is not None:
                    for op_name in sorted(
                        states_group.keys(), key=lambda x: int(x.split("_")[1])
                    ):
                        state_bytes = states_group[op_name][0]
                        entry = pickle.loads(
                            state_bytes.tobytes()
                            if isinstance(state_bytes, np.void)
                            else state_bytes
                        )
                        if system:
                            entry[0]._set_system_(system)
                        states_and_times.append(entry)
                    states_and_times = sorted(
                        states_and_times, key=lambda entry: entry[1]
                    )
                    states = [entry[0] for entry in states_and_times]
                    # The order must be the same...
                    assert time_span == [entry[1] for entry in states_and_times]
                else:
                    states = []

            return cls(parameters, stats, time_span, expect_ops, states)

        except (
            FileNotFoundError,
            OSError,
            PermissionError,
        ) as exc:
            logging.warning(
                "File %s not found. Simulation object could not be loaded (%s).",
                filename,
                str(exc),
            )
            return None
