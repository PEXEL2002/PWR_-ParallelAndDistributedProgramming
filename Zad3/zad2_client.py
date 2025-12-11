import socket
import pickle
import numpy as np
from PIL import Image
import argparse

# python zad2_client.py --host 127.0.0.1 --port 2040


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

def sobel_filter_local(image_array):

    Kx = np.array([[1, 0, -1],
                   [2, 0, -2],
                   [1, 0, -1]])
    Ky = np.array([[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]])


    padded = np.pad(image_array, pad_width=1, mode='edge')
    rows, cols = image_array.shape
    result = np.zeros((rows, cols))

    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            gx = np.sum(Kx * padded[i-1:i+2, j-1:j+2])
            gy = np.sum(Ky * padded[i-1:i+2, j-1:j+2])
            result[i-1, j-1] = np.sqrt(gx**2 + gy**2)

    result = (result / np.max(result) * 255).astype(np.uint8)
    return result

def client_main(host, port):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((host, port))
    print(f"Połączono z serwerem {host}:{port}")

    fragment_info = receive_all(client_socket)
    if fragment_info is None:
        print("Brak danych od serwera.")
        client_socket.close()
        return


    frag = fragment_info['data']
    frag = np.array(frag)

    processed = sobel_filter_local(frag)


    response = {
        'index': fragment_info['index'],
        'start': fragment_info['start'],
        'end': fragment_info['end'],
        'pad_start': fragment_info['pad_start'],
        'pad_end': fragment_info['pad_end'],
        'processed': processed
    }

    send_all(client_socket, response)
    client_socket.close()
    print("Fragment przetworzony i wysłany z powrotem do serwera.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Client for distributed image processing")
    parser.add_argument("--host", required=True, help="Server IP/hostname")
    parser.add_argument("--port", type=int, default=2040, help="Server port")
    args = parser.parse_args()
    client_main(args.host, args.port)