from __future__ import annotations

import socket


class Protocol:
    HEADER = b"\xaa\xaa"
    EOF = b"\xa5\xa5"
    FRAME_LEN_BYTES = 4

    @classmethod
    def wrap_payload_into_frame(cls, payload: bytes) -> bytes:
        # frame = HEADER + frame_len_hex + payload + EOF
        frame_len = len(cls.HEADER) + cls.FRAME_LEN_BYTES + len(payload) + len(cls.EOF)
        frame_len_hex = int.to_bytes(
            frame_len, length=cls.FRAME_LEN_BYTES,
            byteorder='big', signed=True)
        frame = cls.HEADER + frame_len_hex + payload + cls.EOF
        return frame

    @classmethod
    def get_payload_from_frame(cls, frame: bytes) -> bytes | None:
        if len(frame) < len(cls.HEADER) + cls.FRAME_LEN_BYTES + len(cls.EOF):
            print("Incorrect frame format!")
            return None

        # get the frame length info stored in the frame
        frame_len = int.from_bytes(
            frame[len(cls.HEADER):len(cls.HEADER) + cls.FRAME_LEN_BYTES],
            byteorder='big', signed=True)
        # if the frame length is correct, get the payload from it
        if len(frame) == frame_len:
            payload = frame[len(cls.HEADER) + cls.FRAME_LEN_BYTES:-len(cls.EOF)]
            return payload
        else:
            print("Incorrect frame length!")
            return None


class Server:
    def __init__(self):
        self._server_socket = None
        self._connection = None

    def connect(self, ip_addr, port):
        self._server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_socket.bind((ip_addr, port))
        self._server_socket.listen()
        self._connection, address = self._server_socket.accept()
        print('Server connected by ', address)

    def disconnect(self):
        if self._connection is not None:
            self._server_socket.close()

    def send(self, buffer: str) -> int:
        buffer_bytes = buffer.encode('ascii')
        frame = Protocol.wrap_payload_into_frame(buffer_bytes)
        return self._connection.send(frame)

    def receive(self, max_len) -> str | None:
        buffer = self._connection.recv(max_len)
        if len(buffer) > 0:
            frame_begin_idx = buffer.find(Protocol.HEADER)
            frame_end_idx = buffer.rfind(Protocol.EOF)
            if frame_begin_idx != -1 and frame_end_idx != -1:
                frame = buffer[frame_begin_idx:frame_end_idx + len(Protocol.EOF)]
                payload = Protocol.get_payload_from_frame(frame)
                payload_str = payload.decode('ascii')
                return payload_str
        return None
