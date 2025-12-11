import socket
import json
import base64
from PIL import Image
import numpy as np
import argparse


def send_json(sock, obj):
    """Wysyła JSON przez TCP z długością na 8 bajtów (big-endian)"""
    data = json.dumps(obj).encode('utf-8')
    sock.sendall(len(data).to_bytes(8, 'big'))
    sock.sendall(data)


def recv_json(sock):
    """Odbiera JSON przez TCP z długością na 8 bajtów"""
    length_bytes = sock.recv(8)
    if not length_bytes:
        return None
    length = int.from_bytes(length_bytes, 'big')
    data = b''
    while len(data) < length:
        packet = sock.recv(4096)
        if not packet:
            break
        data += packet
    return json.loads(data.decode('utf-8'))


def split_image(image: Image.Image, n_clients: int, overlap: int = 1):
    """Dzieli obraz na fragmenty z zachowaniem overlap, zwraca listę słowników z fragmentami"""
    arr = np.array(image.convert("L"))
    h, w = arr.shape
    n_clients = min(n_clients, h)  # upewniamy się, że n_clients <= wysokość obrazu
    base = h // n_clients
    rem = h % n_clients

    fragments = []
    for i in range(n_clients):
        start = i * base
        end = start + base
        if i == n_clients - 1:
            end += rem

        pad_start = max(0, start - overlap)
        pad_end = min(h, end + overlap)
        if pad_end <= pad_start:
            pad_end = pad_start + 1  # minimalny fragment

        frag = arr[pad_start:pad_end, :].copy()

        # debug print
        print(f"Fragment {i}: start={start}, end={end}, pad_start={pad_start}, pad_end={pad_end}, shape={frag.shape}")

        fragments.append({
            "index": i,
            "start": start,
            "end": end,
            "pad_start": pad_start,
            "pad_end": pad_end,
            "shape": [frag.shape[0], frag.shape[1]],
            "data": base64.b64encode(frag.tobytes()).decode("utf-8")
        })
    return fragments, (h, w)


def merge_image(processed_list, shape):
    """Scala przetworzone fragmenty w jeden obraz"""
    h, w = shape
    result = np.zeros((h, w), dtype=np.uint8)

    processed_list = sorted(processed_list, key=lambda x: x["index"])

    for item in processed_list:
        start = item["start"]
        end = item["end"]
        pad_start = item["pad_start"]
        pad_end = item["pad_end"]

        raw = base64.b64decode(item["processed"])
        arr = np.frombuffer(raw, dtype=np.uint8).reshape((pad_end - pad_start, w))

        top_trim = start - pad_start
        bottom_trim = pad_end - end
        trimmed = arr[top_trim:-bottom_trim, :] if bottom_trim > 0 else arr[top_trim:, :]

        result[start:end, :] = trimmed

    return Image.fromarray(result)


def server_main(image_path, n_clients, host, port, overlap=1):
    image = Image.open(image_path)
    fragments, shape = split_image(image, n_clients, overlap=overlap)

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(n_clients)
    print(f"Serwer nasłuchuje na {host}:{port}, oczekuję {n_clients} klientów...")

    processed_fragments = []

    try:
        for i in range(n_clients):
            client_sock, client_addr = server_socket.accept()
            print(f"Połączono z klientem {i + 1}: {client_addr}")

            send_json(client_sock, fragments[i])
            processed = recv_json(client_sock)
            if processed is None:
                print(f"Błąd: klient {client_addr} zamknął połączenie.")
                client_sock.close()
                continue

            # walidacja przetworzonych danych
            if "processed" not in processed or not processed["processed"]:
                print(f"Błąd: klient {client_addr} wysłał pusty fragment!")
                client_sock.close()
                continue

            processed_fragments.append(processed)
            client_sock.close()
            print(f"Otrzymano przetworzony fragment od klienta {client_addr}")

    finally:
        server_socket.close()

    result_image = merge_image(processed_fragments, shape)
    out_name = "processed_image.png"
    result_image.save(out_name)
    print(f"✅ Obraz przetworzony i zapisany jako {out_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Server for distributed image processing (JSON/Base64)")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--clients", type=int, required=True, help="Number of clients expected")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=2040, help="Port to bind")
    parser.add_argument("--overlap", type=int, default=1, help="Overlap rows for filter")
    args = parser.parse_args()

    server_main(args.image, args.clients, args.host, args.port, overlap=args.overlap)