using System;
using System.Net.Sockets;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading.Tasks;

namespace SobelClientApp
{
    public class FragmentInfo
    {
        [JsonPropertyName("index")]
        public int Index { get; set; }

        [JsonPropertyName("start")]
        public int Start { get; set; }

        [JsonPropertyName("end")]
        public int End { get; set; }

        [JsonPropertyName("pad_start")]
        public int Pad_Start { get; set; }

        [JsonPropertyName("pad_end")]
        public int Pad_End { get; set; }

        [JsonPropertyName("shape")]
        public int[] Shape { get; set; }

        [JsonPropertyName("data")]
        public string Data { get; set; }
    }

    public class ProcessedFragment
    {
        [JsonPropertyName("index")]
        public int Index { get; set; }

        [JsonPropertyName("start")]
        public int Start { get; set; }

        [JsonPropertyName("end")]
        public int End { get; set; }

        [JsonPropertyName("pad_start")]
        public int Pad_Start { get; set; }

        [JsonPropertyName("pad_end")]
        public int Pad_End { get; set; }

        [JsonPropertyName("processed")]
        public string Processed { get; set; }
    }

    class SobelClient
    {
        static async Task Main(string[] args)
        {


            string host = "10.54.153.59";
            int port = 2040;

            try
            {
                using var client = new TcpClient();
                Console.WriteLine($"Łączenie z serwerem {host}:{port}...");
                await client.ConnectAsync(host, port);
                using var stream = client.GetStream();
                
                byte[] lengthBytes = new byte[8];
                await stream.ReadAsync(lengthBytes);
                if (BitConverter.IsLittleEndian)
                    Array.Reverse(lengthBytes);
                long length = BitConverter.ToInt64(lengthBytes, 0);

                Console.WriteLine($"Oczekiwana długość JSON: {length} bajtów");

                // --- Odbiór danych JSON ---
                byte[] buffer = new byte[length];
                int received = 0;
                while (received < length)
                {
                    int bytes = await stream.ReadAsync(buffer, received, (int)length - received);
                    if (bytes == 0) break;
                    received += bytes;
                }

                string json = Encoding.UTF8.GetString(buffer);
                Console.WriteLine($"Otrzymano {received} bajtów JSON");
                
                var options = new JsonSerializerOptions
                {
                    PropertyNameCaseInsensitive = true
                };
                FragmentInfo fragment = JsonSerializer.Deserialize<FragmentInfo>(json, options);
                if (fragment == null)
                {
                    Console.WriteLine("Błąd: odebrany fragment jest null.");
                    return;
                }

                Console.WriteLine($"Odebrano fragment obrazu: {fragment.Index}");
                Console.WriteLine($"Start: {fragment.Start}, End: {fragment.End}");
                Console.WriteLine($"Pad_Start: {fragment.Pad_Start}, Pad_End: {fragment.Pad_End}");
                Console.WriteLine($"Shape: [{fragment.Shape[0]}, {fragment.Shape[1]}]");

                if (string.IsNullOrEmpty(fragment.Data))
                {
                    Console.WriteLine("Błąd: fragment.Data jest pusty!");
                    return;
                }

                Console.WriteLine($"Długość danych Base64: {fragment.Data.Length}");
                
                byte[] raw = Convert.FromBase64String(fragment.Data);

                int rows = fragment.Shape[0];
                int cols = fragment.Shape[1];
                if (raw.Length != rows * cols)
                {
                    Console.WriteLine($"Błąd: rozmiar danych ({raw.Length}) nie pasuje do shape fragmentu ({rows * cols}).");
                    return;
                }

                byte[,] imageArray = new byte[rows, cols];
                for (int i = 0; i < rows; i++)
                    for (int j = 0; j < cols; j++)
                        imageArray[i, j] = raw[i * cols + j];

                Console.WriteLine("Stosowanie filtru Sobela...");

                // --- Zastosowanie filtru Sobela ---
                byte[,] processed = ApplySobel(imageArray, rows, cols);

                // --- Konwersja z powrotem na Base64 ---
                byte[] processedBytes = new byte[rows * cols];
                for (int i = 0; i < rows; i++)
                    for (int j = 0; j < cols; j++)
                        processedBytes[i * cols + j] = processed[i, j];

                ProcessedFragment processedFragment = new ProcessedFragment
                {
                    Index = fragment.Index,
                    Start = fragment.Start,
                    End = fragment.End,
                    Pad_Start = fragment.Pad_Start,
                    Pad_End = fragment.Pad_End,
                    Processed = Convert.ToBase64String(processedBytes)
                };
                
                string processedJson = JsonSerializer.Serialize(processedFragment);
                Console.WriteLine($"Wysyłam JSON: {processedJson.Substring(0, Math.Min(200, processedJson.Length))}...");

                byte[] processedData = Encoding.UTF8.GetBytes(processedJson);
                long dataLength = processedData.Length;
                byte[] processedLength = BitConverter.GetBytes(dataLength);
                if (BitConverter.IsLittleEndian)
                    Array.Reverse(processedLength);

                await stream.WriteAsync(processedLength);
                await stream.WriteAsync(processedData);

                Console.WriteLine($"Fragment przetworzony i wysłany ({dataLength} bajtów) z powrotem do serwera.");
            }
            catch (Exception ex)
            {
                Console.WriteLine("Błąd w kliencie: " + ex);
            }
        }
        
        static byte[,] ApplySobel(byte[,] image, int rows, int cols)
        {
            int[,] Kx = new int[,] { { 1, 0, -1 }, { 2, 0, -2 }, { 1, 0, -1 } };
            int[,] Ky = new int[,] { { 1, 2, 1 }, { 0, 0, 0 }, { -1, -2, -1 } };

            byte[,] result = new byte[rows, cols];
            byte[,] padded = new byte[rows + 2, cols + 2];
            
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    padded[i + 1, j + 1] = image[i, j];

            for (int i = 0; i < cols; i++)
            {
                padded[0, i + 1] = image[0, i];
                padded[rows + 1, i + 1] = image[rows - 1, i];
            }
            for (int i = 0; i < rows; i++)
            {
                padded[i + 1, 0] = image[i, 0];
                padded[i + 1, cols + 1] = image[i, cols - 1];
            }

         
            for (int i = 1; i <= rows; i++)
            {
                for (int j = 1; j <= cols; j++)
                {
                    int gx = 0, gy = 0;
                    for (int ki = -1; ki <= 1; ki++)
                        for (int kj = -1; kj <= 1; kj++)
                        {
                            gx += Kx[ki + 1, kj + 1] * padded[i + ki, j + kj];
                            gy += Ky[ki + 1, kj + 1] * padded[i + ki, j + kj];
                        }
                    int g = (int)Math.Sqrt(gx * gx + gy * gy);
                    if (g > 255) g = 255;
                    result[i - 1, j - 1] = (byte)g;
                }
            }

            return result;
        }
    }
}