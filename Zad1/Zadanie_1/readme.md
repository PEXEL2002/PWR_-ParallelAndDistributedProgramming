# Programowanie równoległe i rozproszone

## Opis projektu

Celem tego projektu jest implementacja symulacji wydobycia węgla przez górników w środowisku równoległym, z wykorzystaniem C#. Projekt ten ma na celu pokazanie, jak liczba równolegle działających wątków (górników) wpływa na czas wykonywania zadania oraz efektywność procesu, uwzględniając problemy związane z dostępem do współdzielonych zasobów.

## Zadania

1. **Symulacja równoległego wydobycia i transportu węgla**
2. **Dodanie podglądu stanu symulacji w czasie rzeczywistym**
3. **Pomiar przyspieszenia i efektywności przy różnych liczbach górników**

Zadania wykonane w ramach tego projektu to:
- Zadanie 1: Implementacja podstawowej symulacji.
- Zadanie 2: Dodanie podglądu statusu symulacji online.
- Zadanie 3: Pomiar efektywności i przyspieszenia przy różnych liczbach górników.

## Wykorzystane technologie

- C#
- Programowanie równoległe
- Klasa `Task` i semafory do synchronizacji wątków

## Instrukcja uruchomienia

Aby uruchomić program, wystarczy skompilować projekt za pomocą Visual Studio lub innego środowiska wspierającego C#. Program działa na systemie Windows.

### Uruchamianie programu

1. **Skopiuj repozytorium na swoje urządzenie.**
2. **Skompiluj projekt w Visual Studio.**
3. **Uruchom program i sprawdź wyniki symulacji.**

## Wyniki

Poniżej przedstawiono przykładowe wyniki symulacji przy różnych liczbach górników:

| Liczba górników | Czas (s)      | Przyspieszenie | Efektywność |
|-----------------|---------------|----------------|-------------|
| 1               | 140,2576437   | 1              | 1           |
| 2               | 72,1934718    | 1,94           | 0,97        |
| 3               | 56,1050136    | 2,50           | 0,83        |
| 4               | 44,1398144    | 3,18           | 0,79        |
| 5               | 36,0809138    | 3,89           | 0,78        |
| 6               | 34,094424     | 4,11           | 0,69        |
