# Zestaw zaliczeniowy — Wybrane Zagadnienia Sztucznej Inteligencji

**Termin oddania:** 27.06.2026r. 
**Forma:** wypełniony Jupyter Notebook + krótkie sprawozdanie w pliku Markdown.

---

## Co znajduje się w folderze

| Plik | Opis |
|------|------|
| `BAI_Zestaw_Zaliczeniowy.ipynb` | Główny notebook — 6 labów × (teoria + przykład + zadanie) |
| `preflight_download.py` | **Uruchom RAZ przed otwarciem notebooka** — sprawdza pakiety i pobiera datasety |
| `README.md` | Ten plik |

---

## Jak zacząć (3 kroki)

```bash
# 1. Aktywuj venv (ten sam co lab_4)
source ../lab_4/.venv/bin/activate

# 2. Doinstaluj pakiety potrzebne zestawowi
pip install datasets scikit-learn jupyter

# 3. Pobierz datasety i sprawdz srodowisko (RAZ, ~5-10 min przy pierwszym razie)
python preflight_download.py

# 4. Otworz notebook
jupyter notebook BAI_Zestaw_Zaliczeniowy.ipynb
```

`preflight_download.py` pobiera 3 datasety z Hugging Face Hub:
- **`zalando-datasets/fashion_mnist`** (~30 MB)
- **`uoft-cs/cifar10`** (~170 MB — to potrwa najdłużej)
- **`dair-ai/emotion`** (~2 MB)

Plus `sklearn iris` (lokalny, bez downloadu).

Wszystko trafia do `~/.cache/huggingface/` — kolejne uruchomienia notebooka są **instant**.

---

## Dane źródłowe — co używasz

| Lab | Dataset | Skąd | Rozmiar (po cache) |
|-----|---------|------|-------|
| 1 | `sklearn iris` | scikit-learn (lokalny) | ~5 KB |
| 1-3, 5 | `zalando-datasets/fashion_mnist` | [Hugging Face](https://huggingface.co/datasets/zalando-datasets/fashion_mnist) | ~30 MB |
| 4-5 | `uoft-cs/cifar10` | [Hugging Face](https://huggingface.co/datasets/uoft-cs/cifar10) | ~170 MB |
| 6 | `dair-ai/emotion` | [Hugging Face](https://huggingface.co/datasets/dair-ai/emotion) | ~2 MB |

**Klucze kolumn (uwaga!):**
- Fashion-MNIST: `ds["train"]["image"]` (zwraca PIL Image)
- CIFAR-10: `ds["train"]["img"]` (**inny klucz!** też PIL)
- Emotion: `ds["train"]["text"]` (string)

---

## Struktura każdej sekcji

Każdy z 6 labów ma identyczną strukturę:

1. **Teoria w trzech zdaniach** — esencja konceptu.
2. **Przykład rozwiązany** — gotowy działający kod z wykresem i wnioskiem.
3. **Twoje zadanie** — analogiczne ćwiczenie z rozszerzonym zakresem (komórka z `# TODO`).

---

## Czas treningu (CPU, M1/M2 Mac)

| Sekcja | Szacowany czas |
|--------|----------------|
| Perceptron (sklearn) | < 5s |
| MLP numpy XOR | < 10s |
| CNN Fashion-MNIST (5 epok) | ~60s |
| CNN CIFAR-10 (8 epok) | ~90s |
| LSTM sinusoida | ~30s |
| LSTM emotion classification | ~3 min |
| Mini grid search (Zadanie 5.1) | ~15 min |

Na GPU wszystko ~5-10x szybsze.

---

## Ocenianie

| Waga | Kryterium |
|------|-----------|
| 50% | Poprawność (czy model się uczy, accuracy > baseline) |
| 30% | Jakość kodu i wizualizacji |
| 20% | Insight — refleksja \"dlaczego ten model się tak zachowuje\" |

---

## FAQ

**Q: Mam tylko CPU, czy zdążę?**
A: Tak. Wszystkie zadania mają subset (5k-10k próbek) i krótkie epoki (5-10).

**Q: Czy mogę użyć PyTorch zamiast TensorFlow?**
A: Tak, ale **uzasadnij wybór** i pamiętaj że examples są w TF — musisz je adaptować sam.

**Q: Czy mogę użyć transfer learning (pretrained ResNet)?**
A: W zadaniu CNN tak — będzie to ciekawy bonus. Ale baseline musisz mieć od zera, żeby porównać.

**Q: Co jeśli HF Hub nie działa / brak internetu?**
A: Po pierwszym `preflight_download.py` wszystko jest **lokalnie w `~/.cache/huggingface/`**. Możesz pracować całkowicie offline.

**Q: Co znaczy \"insight\"?**
A: Coś czego nie ma w PDF-ie laboratoriów. Np. \"Pullover vs Coat ma podobne kontury, więc perceptron myli je w 30% przypadków, ale LogisticRegression z większą elastycznością tnie błąd do 15%\" — to jest insight.
