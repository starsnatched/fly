"""msgpack-RPC address: host + port (AirSim uses TCP 41451)."""
from __future__ import annotations


class Address:
    def __init__(self, ip: str, port: int):
        self.ip = ip
        self.port = int(port)

    def __repr__(self) -> str:
        return f"Address({self.ip!r}, {self.port})"
