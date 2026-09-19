"""Synchronous, thread-safe msgpack-RPC client speaking AirSim's exact wire
format ([0, msgid, method, args] -> [1, msgid, error, result]).

A dedicated reader thread routes responses to waiting callers via msgid, so
sync and async calls can interleave safely on one connection. Object args
(airsim's MsgpackMixin types) are serialized through their to_msgpack()
method, matching the abandoned msgpack-rpc-python behavior.
"""
from __future__ import annotations

import concurrent.futures
import socket
import threading

import msgpack

from .address import Address

# msgpack-RPC message types
REQUEST = 0
RESPONSE = 1


class RPCError(RuntimeError):
    pass


class RPCTimeout(RPCError):
    pass


def _default(obj):
    """msgpack `default` hook: MsgpackMixin objects and numpy scalars."""
    to_msgpack = getattr(obj, "to_msgpack", None)
    if callable(to_msgpack):
        return to_msgpack()
    item = getattr(obj, "item", None)  # numpy scalars
    if callable(item):
        return item()
    raise TypeError(f"cannot serialize {type(obj)!r}")


class Client:
    """One TCP connection per client; `call` is synchronous and thread-safe.

    timeout is in seconds (AirSim passes 3600 = effectively no timeout).
    pack_encoding / unpack_encoding are accepted for API parity with the
    upstream client and ignored.
    """

    def __init__(self, address: Address, timeout: float = 3600.0,
                 pack_encoding: str = "utf-8", unpack_encoding: str = "utf-8"):
        del pack_encoding, unpack_encoding
        if not isinstance(address, Address):
            raise TypeError(f"expected Address, got {type(address)!r}")
        self._addr = address
        self._timeout = float(timeout)
        self._send_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._msgid = 0
        self._sock: socket.socket | None = None
        self._closed = False
        self._pending: dict[int, concurrent.futures.Future] = {}

    # -- connection management -------------------------------------------
    def _ensure_socket(self) -> socket.socket:
        with self._state_lock:
            if self._sock is not None:
                return self._sock
            sock = socket.create_connection((self._addr.ip, self._addr.port),
                                            timeout=10.0)
            sock.settimeout(None)  # reader thread blocks; futures own timeouts
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self._sock = sock
            threading.Thread(target=self._read_loop, args=(sock,),
                             daemon=True).start()
            return sock

    def close(self) -> None:
        with self._state_lock:
            self._closed = True
            sock, self._sock = self._sock, None
            pending, self._pending = self._pending, {}
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass
        for fut in pending.values():
            if not fut.done():
                fut.set_exception(RPCError("connection closed"))

    # -- reader ------------------------------------------------------------
    def _read_loop(self, sock: socket.socket) -> None:
        unpacker = msgpack.Unpacker(raw=False, strict_map_key=False)
        try:
            while True:
                buf = sock.recv(65536)
                if not buf:
                    break
                unpacker.feed(buf)
                for msg in unpacker:
                    if (isinstance(msg, (list, tuple)) and len(msg) >= 4
                            and msg[0] == RESPONSE):
                        fut = self._pending.pop(msg[1], None)
                        if fut is not None and not fut.done():
                            if msg[2] is not None:
                                fut.set_exception(RPCError(str(msg[2])))
                            else:
                                fut.set_result(msg[3])
        except OSError:
            pass
        finally:
            # fail anything still outstanding
            with self._state_lock:
                pending, self._pending = self._pending, {}
                if self._sock is sock:
                    self._sock = None
            for fut in pending.values():
                if not fut.done():
                    fut.set_exception(RPCError("connection closed by server"))

    # -- protocol ----------------------------------------------------------
    def call(self, method: str, *args):
        with self._state_lock:
            if self._closed:
                raise RPCError("client is closed")
            self._msgid += 1
            msgid = self._msgid
            fut: concurrent.futures.Future = concurrent.futures.Future()
            self._pending[msgid] = fut
        payload = msgpack.packb([REQUEST, msgid, method, list(args)],
                                use_bin_type=True, default=_default)
        try:
            sock = self._ensure_socket()
            with self._send_lock:
                _send_all(sock, payload)
            return fut.result(timeout=self._timeout)
        except concurrent.futures.TimeoutError:
            raise RPCTimeout(
                f"{method}: timed out after {self._timeout}s") from None
        except (OSError, ConnectionError) as exc:
            raise RPCError(f"{method}: connection lost ({exc})") from exc

    def call_async(self, method: str, *args):
        """Run `call` in a background thread; returns a Future-like object
        with .join() (parity with msgpack-rpc-python's futures)."""
        fut: concurrent.futures.Future = concurrent.futures.Future()

        def _run():
            try:
                fut.set_result(self.call(method, *args))
            except BaseException as exc:  # propagate to .join()
                fut.set_exception(exc)

        threading.Thread(target=_run, daemon=True).start()

        class _FutureWrapper:
            def join(self):
                return fut.result()

            def __getattr__(self, name):
                return getattr(fut, name)

        return _FutureWrapper()


def _send_all(sock: socket.socket, data: bytes) -> None:
    view = memoryview(data)
    while view:
        sent = sock.send(view)
        view = view[sent:]
