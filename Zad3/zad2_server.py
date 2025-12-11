import socket
import pickle
from PIL import Image
import numpy as np
import argparse
# python zad2_server.py --image images.jpg --clients 3 --host 127.0.0.1 --port 2040


def send_all(sock, obj):
    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    length = len(data)

    sock.sendall(length.to_bytes(8, byteorder='big'))
    sock.sendall(data)

def receive_all(sock):
    length_bytes = sock.recv(8)
    if not length_bytes:
        return None
    length = int.from_bytes(length_bytes, byteorder='big')
    data = b''
    while len(data) < length:
        packet = sock.recv(4096)
        if not packet:
            break
        data += packet
    return pickle.loads(data)

def split_image(image: Image.Image, n_clients: int, overlap: int = 1):

    arr = np.array(image.convert("L"))
    h, w = arr.shape
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

        frag = arr[pad_start:pad_end, :].copy()
        fragments.append({
            'index': i,
            'start': start,
            'end': end,
            'pad_start': pad_start,
            'pad_end': pad_end,
            'data': frag
        })
    return fragments, (h, w)

def merge_image(processed_list, shape):
    h, w = shape
    result = np.zeros((h, w), dtype=np.uint8)


    processed_list = sorted(processed_list, key=lambda x: x['index'])

    for item in processed_list:
        start = item['start']
        end = item['end']
        pad_start = item['pad_start']
        pad_end = item['pad_end']
        proc = item['processed']

        top_trim = start - pad_start
        bottom_trim = pad_end - end

        # wytnij
        if bottom_trim > 0:
            trimmed = proc[top_trim:-bottom_trim, :]
        else:
            trimmed = proc[top_trim:, :]

        # wstaw do rezultatu
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
            print(f"Połączono z klientem {i+1}: {client_addr}")

            send_all(client_sock, fragments[i])

            processed = receive_all(client_sock)
            if processed is None:
                print(f"Błąd: klient {client_addr} zamknął połączenie przed wysłaniem danych.")
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
    print(f"Obraz przetworzony i zapisany jako {out_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Server for distributed image processing")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--clients", type=int, required=True, help="Number of clients expected")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind (default 0.0.0.0)")
    parser.add_argument("--port", type=int, default=2040, help="Port to bind (default 2040)")
    parser.add_argument("--overlap", type=int, default=1, help="Overlap rows for filter (default 1)")
    args = parser.parse_args()

    server_main(args.image, args.clients, args.host, args.port, overlap=args.overlap)
