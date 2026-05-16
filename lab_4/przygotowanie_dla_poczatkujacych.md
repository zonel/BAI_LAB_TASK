# Lab 4 — Przygotowanie dla początkujących

> Ten dokument jest **opcjonalny**. Jeśli czujesz się pewnie z notebookiem — pomiń. Jeśli pierwszy raz dotykasz TensorFlow lub CNN — przeczytaj te 10 minut **przed** otwarciem `Lab4_TensorFlow_CNN_MAIN_FILE.ipynb`.

---

## Co dzisiaj robimy w jednym zdaniu

> Uczymy komputer **rozpoznawać cyfry** na obrazkach — używając frameworka **TensorFlow** zamiast pisać sieć od zera jak na Lab 3.

---

## Dlaczego potrzebujemy TensorFlow

Na Lab 3 napisałeś sieć neuronową **od zera** — własne `forward()`, własny `backpropagation()`, własna pętla treningowa. Działało, ale:

- **50+ linii kodu** dla małej sieci 2-2-1
- **Sam liczyłeś gradienty** z reguły łańcuchowej
- Na sieci 10-warstwowej zwariowałbyś

**TensorFlow** to biblioteka, która robi to wszystko za Ciebie. Mówisz "chcę warstwę 128 neuronów z ReLU" — framework sam dolicza gradienty i aktualizuje wagi.

Można to porównać do nauki:
- **Lab 3** = nauka jazdy na rowerze. Wiesz dokładnie, co robią pedały i przerzutki.
- **Lab 4** = jazda samochodem. Nie musisz wiedzieć, jak działa silnik — wystarczy że umiesz prowadzić.

Oba są ważne. Bez Lab 3 nie zrozumiałbyś, **co robi** TensorFlow w środku.

---

## Co to jest "tensor"?

Tensor to **wielowymiarowa tabelka liczb**. Tyle.

- Skalar = jedna liczba (`5`)
- Wektor = lista liczb (`[1, 2, 3]`)
- Macierz = tabela 2D (`[[1, 2], [3, 4]]`)
- Tensor 3D = sześcian liczb (np. obraz RGB: szerokość × wysokość × 3 kanały)
- Tensor 4D+ = wiele 3D tensorów (np. batch obrazów)

Nie ma w tym nic magicznego. **"Tensor" to ładna nazwa dla wielowymiarowej tablicy numpy.**

---

## Dlaczego CNN, a nie MLP?

MLP (Lab 3) widzi obraz jako **wektor 784 pikseli**. Dla niego piksel (5, 10) i piksel (5, 11) **nie są sąsiednie** — to po prostu dwie różne liczby na pozycji 150 i 151 wektora.

CNN **wie**, że obraz to siatka 2D. Zachowuje informację o tym, co jest blisko czego. Dlatego CNN:

- Rozpoznaje **krawędzie**, **tekstury**, **kształty**
- Działa świetnie na **prawdziwych zdjęciach** (nie tylko MNIST)
- Jest stabilna na **przesunięcia** (cyfra przesunięta o 2 piksele to nadal ta sama cyfra)

To dlatego cały współczesny computer vision (rozpoznawanie twarzy, samochody autonomiczne, medical imaging) opiera się na CNN i jej potomkach.

---

## Cztery cegiełki CNN — w 4 zdaniach

| Cegiełka | Co robi | Analogia |
|----------|--------|----------|
| **Conv2D** | Przesuwa mały filtr po obrazie i wyciąga cechy (krawędzie, tekstury) | Patrzysz przez okienko 3x3 piksele i notujesz, co tam jest |
| **MaxPool2D** | Zmniejsza obraz 2x, biorąc maksymalną wartość z każdego regionu 2x2 | Streszczasz akapit — bierzesz najważniejsze słowo |
| **Flatten** | Zamienia 2D wynik na 1D wektor | Składasz kartkę papieru w linię |
| **Dense** | Zwykła warstwa neuronów (jak w MLP) | Klasyczna sieć — bierze cechy i decyduje "to cyfra 7" |

**Cała CNN** to po prostu: Conv2D → Pool → Conv2D → Pool → Flatten → Dense → Output.

---

## Co to są te "filtry" w Conv2D?

Filtr (kernel) to **mała macierz** — zwykle 3x3 albo 5x5. Przesuwasz ją po obrazie i w każdej pozycji **mnożysz** wartości filtra z wartościami obrazu, **sumujesz**, dostajesz **jedną liczbę**.

Robisz to dla całego obrazu — dostajesz **feature map** (mapę cech).

Przykład: filtr `[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]` (Sobel) **wykrywa pionowe krawędzie**. Gdziekolwiek w obrazie jest pionowa krawędź, na mapie cech dostaniesz wysoką wartość.

**KLUCZ:** w CNN **nie definiujesz filtrów ręcznie**. Sieć sama uczy się, jakich filtrów potrzebuje! W warstwie `Conv2D(24, kernel_size=3)` jest **24 filtrów 3x3**, każdy uczy się wykrywać coś innego.

---

## Po co pooling?

Po konwolucji obraz nadal jest duży. Pooling **redukuje rozmiar**:

```
Wejście 4x4:        Po max pool 2x2:
[[2, 2, 7, 3],      [[9, 7],
 [9, 4, 6, 1],       [8, 6]]
 [8, 5, 2, 4],
 [3, 1, 2, 6]]
```

Dlaczego maksimum? Bo wysokie wartości = silne cechy. Bierzemy najmocniejszy sygnał z każdego regionu.

**Co to daje:**
- Mniej obliczeń (mniejszy obraz → szybszy trening)
- Mniej overfittingu (sieć nie zapamiętuje konkretnej pozycji piksela)
- Niezależność od małych przesunięć

---

## Co to softmax?

Softmax to funkcja aktywacji, której używamy **tylko na warstwie wyjściowej** w klasyfikacji wieloklasowej.

Wejście (10 liczb): `[2.0, 1.0, 0.1, ...]`
Wyjście (10 liczb sumujących się do 1): `[0.7, 0.2, 0.05, ...]`

Mówiąc inaczej: zamienia "logity" (surowe wyniki) na **prawdopodobieństwa**. Jeśli wyjście to `[0.001, 0.002, ..., 0.97, ..., 0.001]`, sieć mówi: "to jest cyfra 7 z 97% pewnością".

---

## Glossariusz angielski → polski

| Po angielsku | Po polsku | Znaczenie |
|--------------|-----------|-----------|
| layer | warstwa | jeden poziom sieci |
| filter / kernel | filtr / jądro | mała macierz do konwolucji |
| feature map | mapa cech | wynik konwolucji |
| stride | krok | jak daleko przesuwamy filtr |
| padding | wypełnienie | zera dodawane na krawędziach |
| pooling | grupowanie | redukcja rozmiaru |
| flatten | spłaszczanie | zamiana 2D → 1D |
| dense / fully-connected | gęsta / w pełni połączona | klasyczna warstwa MLP |
| epoch | epoka | jedno przejście przez cały zbiór treningowy |
| batch | partia | grupka próbek, po której aktualizujemy wagi |
| loss | strata | jak bardzo sieć się myli |
| accuracy | dokładność | % poprawnych predykcji |
| optimizer | optymalizator | algorytm aktualizujący wagi (np. Adam, SGD) |
| overfitting | przeuczenie | sieć zna train na pamięć, ale fails na test |

---

## Najczęstsze problemy i ich rozwiązania

### Problem: `ModuleNotFoundError: No module named 'tensorflow'`

```bash
pip install tensorflow
```

Pierwsza instalacja waży ~600 MB. Cierpliwie.

### Problem: trening trwa wieczność

CNN jest 10-100x wolniejsza od MLP. Na CPU 10 epok = kilka minut. Jeśli masz GPU (NVIDIA) — TensorFlow automatycznie go użyje. Sprawdź:

```python
tf.config.list_physical_devices('GPU')
```

### Problem: `accuracy` stoi w miejscu na ~10%

10% = losowanie (10 klas → 1/10 szans). Sprawdź:
- Czy znormalizowałeś dane? `x_train / 255.0`
- Czy etykiety są one-hot? `tf.keras.utils.to_categorical(y_train, 10)`
- Czy dane mają właściwy shape? Dla CNN: `(N, 28, 28, 1)`, nie `(N, 784)`

### Problem: `val_accuracy` rośnie, potem zaczyna spadać

To **overfitting**. Sieć uczy się zbyt dobrze danych treningowych, a gorzej generalizuje. Rozwiązania:
- Dodaj `Dropout(0.5)` między warstwami
- Dodaj `kernel_regularizer=tf.keras.regularizers.l2(0.01)`
- Zmniejsz model (mniej neuronów)
- Augmentacja danych

### Problem: `ValueError: expected min_ndim=4, found ndim=3`

CNN wymaga 4D tensora: `(batch, height, width, channels)`. Musisz dodać wymiar kanałów:

```python
x_train = x_train.reshape(-1, 28, 28, 1)   # dla obrazów grayscale
```

---

## Quick reference: jak zbudować CNN w TensorFlow

```python
import tensorflow as tf

# 1. Wczytanie danych
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 2. Preprocessing
x_train = x_train.reshape(-1, 28, 28, 1) / 255.0  # 4D + normalizacja
x_test = x_test.reshape(-1, 28, 28, 1) / 255.0
y_train = tf.keras.utils.to_categorical(y_train, 10)  # one-hot
y_test = tf.keras.utils.to_categorical(y_test, 10)

# 3. Architektura
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(24, 3, activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPool2D(2),
    tf.keras.layers.Conv2D(36, 3, activation='relu'),
    tf.keras.layers.MaxPool2D(2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# 4. Kompilacja
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 5. Trening
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# 6. Ewaluacja
loss, acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {acc:.4f}")
```

To wszystko. **15 linii** = działająca CNN.

---

## Co czytać dalej

- **3Blue1Brown** — *But what is a neural network?* (YouTube, najlepsza wizualna intuicja)
- **CS231n Stanford** — *Convolutional Neural Networks for Visual Recognition* (najlepszy kurs ML free online)
- **TensorFlow tutorials** — `tensorflow.org/tutorials` (oficjalne, bardzo dobre)
- **fast.ai** — kurs deep learningu "from top to bottom" (najszybsza droga do produkcji)

Powodzenia. Otwórz teraz notebook i jedź komórka po komórce.
