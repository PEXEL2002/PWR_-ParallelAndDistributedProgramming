// Program.cs
using System;
using System.Diagnostics;
using System.Threading;
using System.Threading.Tasks;

namespace Zadanie_1
{
    class Program
    {
        // =====================================================================
        // ▒▒▒ PARAMETRY SYMULACJI ▒▒▒
        // =====================================================================

        const int ROZMIAR_ZLOZA = 2000;   // jednostek węgla
        const int POJEMNOSC_POJAZDU = 200;    // jednostek
        const int MS_WYDOBYCIE_1 = 10;     // czas wydobycia 1 jednostki [ms]
        const int MS_ROZLADUNEK_1 = 10;     // czas rozładunku 1 jednostki [ms]
        const int MS_PRZEJAZD = 10_000; // czas przejazdu w obie strony [ms]

        // Skalowanie czasu (1.0 = normalny, <1 = szybszy)
        const double TIME_SCALE = 1.0;

        // =====================================================================
        // ▒▒▒ STAN WSPÓŁDZIELONY ▒▒▒
        // =====================================================================

        static int zlozePozostalo = ROZMIAR_ZLOZA;
        static int dostarczono = 0;

        static readonly object lockZloze = new object();
        static readonly SemaphoreSlim semZloze = new SemaphoreSlim(2, 2);   // max 2 górników przy złożu
        static readonly SemaphoreSlim semMagazyn = new SemaphoreSlim(1, 1);
        static float def_time = 0;
        // tylko 1 pojazd w magazynie

        // =====================================================================
        // ▒▒▒ FUNKCJE POMOCNICZE ▒▒▒
        // =====================================================================

        /// <summary>
        /// Uśpienie z uwzględnieniem współczynnika TIME_SCALE.
        /// </summary>
        static void SleepScaled(int ms)
        {
            int scaled = (int)Math.Max(1, ms * TIME_SCALE);
            Thread.Sleep(scaled);
        }

        /// <summary>
        /// Aktualizacja statusu w konsoli z zachowaniem pozycji kursora.
        /// </summary>
        static void WypiszStatus(int wiersz, string tekst)
        {
            lock (Console.Out) // ochrona przed mieszaniem się tekstów
            {
                Console.SetCursorPosition(0, wiersz);
                Console.WriteLine(tekst.PadRight(60));
            }
        }

        // =====================================================================
        // ▒▒▒ GŁÓWNA PĘTLA GÓRNIKA ▒▒▒
        // =====================================================================
        static void PetlaGornika(int id)
        {
            while (true)
            {
                // --- Wejście do złoża (max 2 górników naraz) ---
                semZloze.Wait();
                int doWydobycia = 0;

                // Rezerwacja porcji surowca
                lock (lockZloze)
                {
                    if (zlozePozostalo > 0)
                    {
                        doWydobycia = Math.Min(POJEMNOSC_POJAZDU, zlozePozostalo);
                        zlozePozostalo -= doWydobycia;
                    }
                }

                if (doWydobycia == 0)
                {
                    // Koniec pracy – brak węgla
                    semZloze.Release();
                    //WypiszStatus(id + 3, $"Górnik {id} zakończył pracę.");
                    return;
                }

                // --- Wydobycie ---
                //WypiszStatus(id + 3, $"Górnik {id}: wydobywa {doWydobycia} jednostek...");
                SleepScaled(MS_WYDOBYCIE_1 * doWydobycia);

                semZloze.Release(); // zwolnij złoże

                // --- Transport ---
                //WypiszStatus(id + 3, $"Górnik {id}: transportuje do magazynu...");
                SleepScaled(MS_PRZEJAZD);

                // --- Rozładunek ---
                semMagazyn.Wait();
                //WypiszStatus(id + 3, $"Górnik {id}: rozładowuje węgiel...");
                SleepScaled(MS_ROZLADUNEK_1 * doWydobycia);
                Interlocked.Add(ref dostarczono, doWydobycia);
                semMagazyn.Release();

                // Aktualizacja statusu złoża i magazynu
                //WypiszStatus(0, $"Stan złoża: {zlozePozostalo}");
                //WypiszStatus(1, $"Stan magazynu: {dostarczono}");
            }
        }

        // =====================================================================
        // ▒▒▒ FUNKCJA GŁÓWNA ▒▒▒
        // =====================================================================
        static public void Badania()
        {
            for (int j = 1; j <= 6; j++)
            {
                Console.WriteLine($"Badanie dla {j} górników");
                zlozePozostalo = ROZMIAR_ZLOZA;
                dostarczono = 0;
                int liczbaGornikow = j;

                //Console.Clear();
                //WypiszStatus(0, $"Stan złoża: {zlozePozostalo}");
                //WypiszStatus(1, $"Stan magazynu: {dostarczono}");
                //Console.WriteLine();

                var stoper = Stopwatch.StartNew();

                // Uruchom wątki górników
                Task[] tasks = new Task[liczbaGornikow];
                for (int i = 0; i < liczbaGornikow; i++)
                {
                    int id = i + 1;
                    tasks[i] = Task.Run(() => PetlaGornika(id));
                }

                Task.WaitAll(tasks);
                stoper.Stop();
                if (j == 1)
                {
                    def_time = (float)stoper.Elapsed.TotalSeconds;
                }
                float przysieszenie = def_time / (float)stoper.Elapsed.TotalSeconds;
                float efektywnosc = przysieszenie / liczbaGornikow;

                Console.WriteLine($"Liczba górników:{liczbaGornikow}, czas: {stoper.Elapsed.TotalSeconds} s, przyśpieszenie: {przysieszenie}, efektywnosc: {efektywnosc} ");
            }
        }
        static void Main(string[] args)
        {
            Badania();
            return;
            int liczbaGornikow = 5;

            if (args.Length >= 1 && int.TryParse(args[0], out var n) && n > 0)
                liczbaGornikow = n;

            Console.Clear();
            WypiszStatus(0, $"Stan złoża: {zlozePozostalo}");
            WypiszStatus(1, $"Stan magazynu: {dostarczono}");
            Console.WriteLine();

            var stoper = Stopwatch.StartNew();

            // Uruchom wątki górników
            Task[] tasks = new Task[liczbaGornikow];
            for (int i = 0; i < liczbaGornikow; i++)
            {
                int id = i + 1;
                tasks[i] = Task.Run(() => PetlaGornika(id));
            }

            Task.WaitAll(tasks);
            stoper.Stop();

            // Podsumowanie
            Console.WriteLine();
            Console.WriteLine("───────────────────────────────────────────────");
            Console.WriteLine($"Zakończono. Dostarczono: {dostarczono} / {ROZMIAR_ZLOZA} jednostek.");
            Console.WriteLine($"Czas całkowity: {stoper.Elapsed}");
            Console.WriteLine("───────────────────────────────────────────────");
        }
        
    }
}
