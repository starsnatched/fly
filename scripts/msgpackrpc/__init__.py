"""Minimal msgpack-RPC client shim (replaces the abandoned msgpack-rpc-python
package, which no longer imports on modern Python).

Speaks the exact wire format AirSim's server expects:
  request  = [msg_type=0, msgid, method(str), args]
  response = [msg_type=1, msgid, error, result]
Synchronous, thread-safe (one socket per Client), no external deps beyond
msgpack itself.
"""
from .client import Client
from .address import Address

__all__ = ["Client", "Address"]
