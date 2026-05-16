# 🧠 Lab 3 — Dopełnienie notebooka

> **Ten dokument jest TOWARZYSZEM do notebooka `Lab3_MLP_Backpropagation_MAIN_FILE.ipynb`.**
>
> Notebook pokazuje **CO** zrobić (kod, wzory, wizualizacje). Ten plik wyjaśnia **DLACZEGO każda linijka wygląda tak, a nie inaczej** — krok po kroku, z każdym kształtem macierzy, każdą operacją, każdym `@`, każdym `outer`, każdym `[:, 1:]`.

Czytaj ten plik **równolegle** z notebookiem. Każda sekcja odnosi się do konkretnej komórki.

---

## 📑 Spis treści

1. [Pochodne sigmoid i tanh — wyprowadzenie krok po kroku](#1) (uzupełnia komórkę 6)
2. [Forward pass — anatomia każdej linii](#2) (uzupełnia komórki 10, 13)
3. [Reguła łańcuchowa — pełne wyprowadzenie gradientów](#3) (uzupełnia komórkę 11)
4. [Backpropagation w kodzie — co znaczy każda linia `train_mlp_xor`](#4) (uzupełnia komórkę 14)
5. [Tablica kształtów — który wymiar gdzie](#5)
6. [Pułapki numeryczne i decyzje projektowe](#6)
7. [Zadanie 1 — rozwiązanie z komentarzem](#7)
8. [Zadanie 2 — rozwiązanie z komentarzem](#8)
9. [Odpowiedzi na pytania kontrolne](#9)

---

<a id="1"></a>

# 1. Pochodne sigmoid i tanh — wyprowadzenie krok po kroku

> **Notebook (komórka 6) podaje wzory pochodnych jako fakt. Tutaj wyprowadzamy je tak, że nie zostaje żadne "dlaczego".**

## 1.1 Pochodna sigmoidy — pełne wyprowadzenie

Notebook mówi: `sigmoid_derivative(y) = β·y·(1-y)`. Skąd to?

### Punkt wyjścia

```
σ(x) = 1 / (1 + e^(-βx))
```

### Zapis pomocniczy

Niech `u = 1 + e^(-βx)`. Wtedy `σ(x) = 1/u = u^(-1)`.

### Krok 1: pochodna złożenia

```
dσ/dx = -1 · u^(-2) · du/dx
      = -1/u² · du/dx
```

### Krok 2: liczymy `du/dx`

```
u = 1 + e^(-βx)
du/dx = e^(-βx) · (-β)
      = -β · e^(-βx)
```

### Krok 3: składamy

```
dσ/dx = -1/u² · (-β · e^(-βx))
      = β · e^(-βx) / u²
      = β · e^(-βx) / (1 + e^(-βx))²
```

### Krok 4: magiczny trik algebraiczny

Rozkładamy ułamek:

```
β · e^(-βx) / (1 + e^(-βx))²
= β · [1 / (1 + e^(-βx))] · [e^(-βx) / (1 + e^(-βx))]
       └────────────────┘   └─────────────────────┘
            σ(x)              ?
```

Drugi czynnik:

```
e^(-βx) / (1 + e^(-βx))
= (1 + e^(-βx) - 1) / (1 + e^(-βx))
= 1 - 1/(1 + e^(-βx))
= 1 - σ(x)
```

### Wynik końcowy

```
σ'(x) = β · σ(x) · (1 - σ(x))
```

🔍 **Dlaczego to jest cudowne?**

Pochodna jest wyrażona przez **samą wartość σ(x)**, którą JUŻ POLICZYŁEŚ w forward pass. Nie musisz powtarzać exp, dzielenia ani niczego. Kosztuje to dosłownie **dwa mnożenia**: `β * y * (1 - y)`.

🔍 **Dlatego w kodzie notebooka argument to `y`, nie `x`:**

```python
def sigmoid_derivative(y, beta=1.0):
    """Pochodna sigmoidy. UWAGA: y to juz WYJSCIE sigmoidy, nie x!"""
    return beta * y * (1 - y)
```

Komentarz "y to już WYJŚCIE sigmoidy" jest kluczowy. Łatwo się pomylić i podać `x` zamiast `y`.

---

## 1.2 Pochodna tanh — pełne wyprowadzenie

### Punkt wyjścia

```
tanh(βx) = (e^(βx) - e^(-βx)) / (e^(βx) + e^(-βx))
```

### Krok 1: tożsamość trygonometryczna

Wiemy z analizy:

```
d/dx [tanh(x)] = 1 - tanh²(x)
```

### Krok 2: dodajemy β (reguła łańcuchowa)

```
d/dx [tanh(βx)] = β · (1 - tanh²(βx))
```

### Wynik

```
tanh'(x) = β · (1 - tanh²(x))
```

🔍 **Znów ten sam pattern:** pochodna wyrażona przez **wartość funkcji**. To dlatego sigmoid i tanh są standardem od dekad.

🔍 **W kodzie:**

```python
def tanh_derivative(y, beta=1.0):
    """Pochodna tanh. UWAGA: y to juz WYJSCIE tanh, nie x!"""
    return beta * (1 - y**2)
```

Tu `y` to wartość po tanh, czyli `v_hidden` z forward pass. Dlatego w `train_mlp_xor` widzisz: `tanh_derivative(v_hidden, beta)`, nie `tanh_derivative(z_hidden, beta)`.

---

## 1.3 Sigmoid vs Tanh — porównanie pochodnych

| Cecha | σ'(x) | tanh'(x) |
|-------|-------|----------|
| Wartość maksymalna | `β/4` w punkcie 0 | `β` w punkcie 0 |
| Wartość na krańcach | → 0 dla `\|x\| > 5` | → 0 dla `\|x\| > 3` |
| Konsekwencja | Słabsze gradienty | Silniejsze gradienty (4× większe) |

🔍 **To jeden z powodów, dla których tanh jest preferowany w warstwach ukrytych.** Większe gradienty = szybsze uczenie. Sigmoid utrzymuje się tylko na wyjściu, gdzie chcemy interpretację `[0, 1]`.

---

<a id="2"></a>

# 2. Forward pass — anatomia każdej linii

> **Notebook (komórka 13) pokazuje funkcję `mlp_forward()`. Tutaj rozkładamy ją linia po linii ze wszystkimi kształtami i decyzjami projektowymi.**

## 2.1 Dlaczego zwracamy aż 5 wartości?

```python
def mlp_forward(x, W1, W2, beta=1.0):
    ...
    return y, v, v_hidden, z_hidden, z_output
```

🔍 **Pytanie:** dlaczego nie tylko `y`?

🔍 **Odpowiedź:** bo backpropagation potrzebuje **wszystkich wartości pośrednich**.

| Zwracane | Do czego potrzebne |
|----------|---------------------|
| `y` | Obliczenie błędu `(y - d)` |
| `v` | Gradient W²: `δ_out · v` |
| `v_hidden` | Pochodna tanh: `tanh'(v_hidden)` |
| `z_hidden` | Debug, przydatne dla wizualizacji |
| `z_output` | Debug, przydatne dla wizualizacji |

Bez tych wartości backpropagation musiałby je przeliczać od nowa = strata mocy obliczeniowej.

---

## 2.2 Linia po linii

### Linia 1: `z_hidden = W1 @ x`

```python
z_hidden = W1 @ x
```

| Element | Kształt | Co robi |
|---------|---------|---------|
| `W1` | `(2, 3)` | macierz wag warstwy ukrytej |
| `x` | `(3,)` | wejście z biasem `[1, x1, x2]` |
| `@` | operator | mnożenie macierzy |
| `z_hidden` | `(2,)` | pre-aktywacja każdego neuronu ukrytego |

🔍 **Co dokładnie liczy `@`?**

```
z_hidden[0] = W1[0,0]*x[0] + W1[0,1]*x[1] + W1[0,2]*x[2]
            = w1₁₀·1 + w1₁₁·x1 + w1₁₂·x2
            = bias_neurona1 + waga11·x1 + waga12·x2

z_hidden[1] = W1[1,0]*x[0] + W1[1,1]*x[1] + W1[1,2]*x[2]
            = bias_neurona2 + waga21·x1 + waga22·x2
```

🔍 **Dlaczego `@`, nie `*`?** `*` to mnożenie element-po-elemencie (Hadamard). Dla `(2,3) * (3,)` numpy zrobiłby broadcasting i wyszłaby macierz, nie wektor. `@` to **mnożenie macierzy**, czyli to czego naprawdę chcemy.

### Linia 2: `v_hidden = tanh_act(z_hidden, beta)`

```python
v_hidden = tanh_act(z_hidden, beta)
```

| Element | Kształt | Co robi |
|---------|---------|---------|
| `z_hidden` | `(2,)` | pre-aktywacja |
| `tanh_act` | element-wise | aplikuje tanh do każdego elementu |
| `v_hidden` | `(2,)` | post-aktywacja, wartości w `(-1, 1)` |

🔍 **Element-wise = każdy element przetworzony niezależnie.** `v_hidden[0]` zależy tylko od `z_hidden[0]`, nie od `z_hidden[1]`. To dlatego pochodna tanh jest skalarna dla każdego elementu osobno.

### Linia 3: `v = np.concatenate([[1], v_hidden])`

```python
v = np.concatenate([[1], v_hidden])
```

| Element | Kształt | Co robi |
|---------|---------|---------|
| `[[1]]` | lista 1-elementowa | bias |
| `v_hidden` | `(2,)` | wyjścia warstwy ukrytej |
| `concatenate` | sklejanie | `[1, v_hidden[0], v_hidden[1]]` |
| `v` | `(3,)` | wejście do warstwy wyjściowej |

🔍 **Dlaczego ręcznie doklejamy `1`?**

W konwencji "bias jako waga" (używana w tym labie) bias to po prostu kolejna waga, ale skojarzona ze stałym wejściem `1`. Doklejając `1` do wektora wejściowego, bias automatycznie zostaje dodany przy mnożeniu macierzowym `W2 @ v`:

```
W2 @ v = W2[0]*1 + W2[1]*v_hidden[0] + W2[2]*v_hidden[1]
       = bias  + waga1*v1            + waga2*v2
```

🔍 **Alternatywa współczesna:** w PyTorch/TensorFlow bias to osobny wektor `b`. Wtedy `z = W·x + b`. Jest to czystsze, ale wymaga więcej kodu. Dla labu konwencja "bias jako waga" jest prostsza dydaktycznie.

### Linia 4: `z_output = W2 @ v`

```python
z_output = W2 @ v
```

| Element | Kształt | Co robi |
|---------|---------|---------|
| `W2` | `(1, 3)` | wagi warstwy wyjściowej |
| `v` | `(3,)` | wejście (z biasem) |
| `z_output` | `(1,)` | pre-aktywacja wyjścia (1-elementowy wektor) |

🔍 **Dlaczego `(1, 3) @ (3,) → (1,)`, a nie skalar?**

Bo `W2` jest **macierzą** o kształcie `(1, 3)`, a nie wektorem `(3,)`. Numpy zachowuje wymiar pierwszy (`1`) w wyniku. To spójna konwencja: liczba neuronów wyjściowych = pierwszy wymiar wyjścia.

🔍 **W generalnej sieci z `M` wyjściami `W2` ma kształt `(M, K+1)` i wynik `z_output` ma kształt `(M,)`.** Tu mamy `M=1`, więc `(1,)`.

### Linia 5: `y = sigmoid(z_output, beta)`

```python
y = sigmoid(z_output, beta)
```

Kształt `y`: `(1,)` (taki sam jak `z_output`, sigmoid też element-wise).

---

## 2.3 Diagram przepływu z kształtami

```
x [shape=(3,)]
│  [1, x1, x2]
│
▼  W1 @ x  ←  W1 [shape=(2,3)]
│
z_hidden [shape=(2,)]
│  [bias+w·x dla neurona 1, bias+w·x dla neurona 2]
│
▼  tanh(β·z)
│
v_hidden [shape=(2,)]
│  [v1, v2] -- każde w (-1, 1)
│
▼  np.concatenate([[1], v_hidden])
│
v [shape=(3,)]
│  [1, v1, v2]
│
▼  W2 @ v  ←  W2 [shape=(1,3)]
│
z_output [shape=(1,)]
│
▼  sigmoid(β·z)
│
y [shape=(1,)]   -- wartość w (0, 1)
```

---

<a id="3"></a>

# 3. Reguła łańcuchowa — pełne wyprowadzenie gradientów

> **Notebook (komórka 11) podaje gotowe wzory na gradienty. Tutaj wyprowadzamy je krok po kroku, żeby było jasne skąd każdy element się bierze.**

## 3.1 Co chcemy

Mamy funkcję błędu:

```
E(W) = (1/2) · Σ_m (y_m - d_m)²
```

Chcemy `∂E/∂w` dla **każdej** wagi w sieci. Dlaczego? Bo wzór aktualizacji to:

```
w_nowe = w_stare - η · (∂E/∂w)
```

W naszej sieci 2-2-1 mamy:
- **6 wag w W1**: 2 neurony × 3 wagi (bias + 2 wejścia)
- **3 wagi w W2**: 1 neuron × 3 wagi (bias + 2 wejścia z warstwy ukrytej)
- **Razem: 9 gradientów do policzenia**

---

## 3.2 Łańcuch zależności

```
w⁽¹⁾_kn  →  z_hidden_k  →  v_hidden_k  →  z_out  →  y  →  E
            (W1@x)         (tanh)         (W2@v)   (sigmoid)  (MSE)

w⁽²⁾_mk  →  z_out  →  y  →  E
            (W2@v)    (sigmoid)
```

Reguła łańcuchowa mówi: **mnożymy pochodne wzdłuż łańcucha**.

---

## 3.3 Gradient dla W² — krok po kroku

Cel: `∂E/∂w⁽²⁾_mk` (waga między m-tym wyjściem a k-tym neuronem ukrytym).

### Łańcuch

```
∂E/∂w⁽²⁾_mk = (∂E/∂y_m) · (∂y_m/∂z_out_m) · (∂z_out_m/∂w⁽²⁾_mk)
```

### Pochodna 1: `∂E/∂y_m`

```
E = (1/2) · Σ_m (y_m - d_m)²

∂E/∂y_m = (1/2) · 2 · (y_m - d_m)
        = (y_m - d_m)
```

🔍 **Dlatego MSE ma `1/2` na początku** — żeby ta pochodna była ładnym `(y - d)` bez dwójki. To kosmetyka algebraiczna, ale wszyscy ją stosują.

### Pochodna 2: `∂y_m/∂z_out_m`

```
y_m = sigmoid(z_out_m, β)

∂y_m/∂z_out_m = β · y_m · (1 - y_m)
```

(z części 1.1)

### Pochodna 3: `∂z_out_m/∂w⁽²⁾_mk`

```
z_out_m = Σ_k w⁽²⁾_mk · v_k

∂z_out_m/∂w⁽²⁾_mk = v_k
```

🔍 **Pochodna sumy ważonej po wadze = wejście tej wagi.** Banalne, ale kluczowe.

### Złożenie

```
∂E/∂w⁽²⁾_mk = (y_m - d_m) · β·y_m·(1 - y_m) · v_k
              └─────────┘   └──────────────┘   └─┘
              błąd          δ_out (lokalny błąd) v_k
```

---

## 3.4 Definicja delty wyjściowej

Wyodrębniamy część "lokalną" do warstwy wyjściowej:

```
δ_out_m = (y_m - d_m) · β · y_m · (1 - y_m)
```

Wtedy:

```
∂E/∂w⁽²⁾_mk = δ_out_m · v_k
```

🔍 **W kodzie notebooka:**

```python
delta_output = error * sigmoid_derivative(y, beta)
#              (y-d)   β·y·(1-y)
#              └────── δ_out ───────┘
```

---

## 3.5 Gradient dla W¹ — krok po kroku

Cel: `∂E/∂w⁽¹⁾_kn` (waga między k-tym neuronem ukrytym a n-tym wejściem).

### Łańcuch (dłuższy!)

```
∂E/∂w⁽¹⁾_kn = Σ_m [(∂E/∂y_m) · (∂y_m/∂z_out_m) · (∂z_out_m/∂v_k)] 
              · (∂v_k/∂z_hidden_k) · (∂z_hidden_k/∂w⁽¹⁾_kn)
```

🔍 **Suma po m, bo każdy neuron warstwy ukrytej wpływa na KAŻDY neuron warstwy wyjściowej.** W naszej sieci `M=1`, więc suma ma 1 element, ale ogólny wzór wymaga sumy.

### Pochodne

| Element | Wartość | Skąd |
|---------|---------|------|
| `∂E/∂y_m` | `(y_m - d_m)` | MSE |
| `∂y_m/∂z_out_m` | `β · y_m · (1 - y_m)` | sigmoid' |
| `∂z_out_m/∂v_k` | `w⁽²⁾_mk` | suma ważona |
| `∂v_k/∂z_hidden_k` | `β · (1 - v_hidden_k²)` | tanh' |
| `∂z_hidden_k/∂w⁽¹⁾_kn` | `x_n` | suma ważona |

### Złożenie

```
∂E/∂w⁽¹⁾_kn = Σ_m [δ_out_m · w⁽²⁾_mk] · β·(1-v_hidden_k²) · x_n
              └─────────────────────┘   └──────────────┘   └─┘
              błąd propagowany wstecz   pochodna tanh      wejście
```

### Definicja delty ukrytej

```
δ_hidden_k = (Σ_m δ_out_m · w⁽²⁾_mk) · β · (1 - v_hidden_k²)
```

Wtedy:

```
∂E/∂w⁽¹⁾_kn = δ_hidden_k · x_n
```

🔍 **Identyczna struktura jak dla W²:** `gradient = δ_warstwy · wejście_z_warstwy_poprzedniej`.

---

## 3.6 Dlaczego to nazywa się "wsteczna" propagacja?

Spójrz na deltę warstwy ukrytej:

```
δ_hidden_k = (Σ_m δ_out_m · w⁽²⁾_mk) · pochodna_aktywacji
```

🔍 **Błąd warstwy ukrytej = ważona suma błędów warstwy wyjściowej.**

Wagi `w⁽²⁾_mk`, którymi sygnał płynął **w przód** (z ukrytej do wyjściowej), są tymi samymi, którymi błąd płynie **wstecz** (z wyjściowej do ukrytej).

Stąd nazwa: **back-propagation** = propagacja błędu **w tył** przez te same wagi.

---

<a id="4"></a>

# 4. Backpropagation w kodzie — co znaczy każda linia `train_mlp_xor`

> **Notebook (komórka 14) zawiera funkcję `train_mlp_xor`. Tutaj rozkładamy KAŻDĄ linię backpropagation.**

## 4.1 Inicjalizacja wag

```python
np.random.seed(42)
W1 = np.random.randn(K, N + 1) * 0.5  # K x (N+1)
W2 = np.random.randn(M, K + 1) * 0.5  # M x (K+1)
```

🔍 **Dlaczego `* 0.5`?**

Bo `np.random.randn` daje wartości z N(0, 1) — czyli ~68% w przedziale `[-1, 1]`. Pomnożone przez 0.5 mamy wagi w przedziale ~`[-0.5, 0.5]`. To **małe losowe wartości**, jak wymaga instrukcja.

🔍 **Dlaczego małe?**

Bo duże wagi → duża pre-aktywacja `z` → tanh i sigmoid w stanie nasycenia → pochodna ~0 → **gradient znika** → sieć się nie uczy. Na początku chcemy działać w środku zakresu, gdzie pochodne są duże.

🔍 **Dlaczego nie zera?**

Bo wszystkie neurony stałyby się identyczne (tzw. **symmetry breaking problem**). Każdy neuron musi być "trochę inny", żeby uczyły się różnych rzeczy. Małe losowe wagi to gwarantują.

---

## 4.2 Akumulatory gradientów

```python
dW1 = np.zeros_like(W1)  # (K, N+1) = (2, 3)
dW2 = np.zeros_like(W2)  # (M, K+1) = (1, 3)
```

🔍 **`np.zeros_like(W1)`** = macierz zer o **identycznym kształcie** jak `W1`. Wygodne — nie musimy hardkodować wymiarów.

🔍 **Dlaczego akumulujemy?** Bo to wariant **batch** (aktualizacja po epoce). Liczymy gradient dla każdej próbki, dodajemy do akumulatora, dopiero pod koniec epoki aktualizujemy wagi.

---

## 4.3 Forward pass dla próbki

```python
x = np.concatenate([[1], X[i]])  # [1, x1, x2]
target = d[i]
y, v, v_hidden, z_hidden, z_output = mlp_forward(x, W1, W2, beta)
```

🔍 **`np.concatenate([[1], X[i]])`** = doklejanie biasu do wejścia. To samo co w forward pass — pamiętasz `[1, x1, x2]`?

---

## 4.4 Obliczenie błędu

```python
error = y - target  # (y - d), shape: (M,)
total_error += 0.5 * np.sum(error**2)
```

🔍 **`error` to wektor**, nie skalar. W naszej sieci ma kształt `(1,)`, ale w ogólnej sieci z `M` wyjściami miałby `(M,)`.

🔍 **`0.5 * np.sum(error**2)`** = MSE dla tej próbki. Sumujemy po wszystkich wyjściach (`M`), mnożymy przez `1/2`.

🔍 **`total_error +=`** — akumulujemy błąd przez całą epokę, żeby narysować krzywą uczenia.

---

## 4.5 Delta warstwy wyjściowej

```python
delta_output = error * sigmoid_derivative(y, beta)  # (M,)
```

🔍 **Element-wise mnożenie** (`*`). `error` ma kształt `(M,)`, `sigmoid_derivative(y)` też `(M,)`, wynik `(M,)`.

🔍 **To dokładnie δ_out z części 3.4:**

```
δ_out_m = (y_m - d_m) · β · y_m · (1 - y_m)
          └─ error ─┘   └─ sigmoid_derivative(y) ─┘
```

---

## 4.6 Gradient W² — KLUCZOWA LINIA

```python
dW2 += np.outer(delta_output, v)  # (M, K+1)
```

🔍 **Co to jest `np.outer`?**

`np.outer(a, b)` to **iloczyn zewnętrzny** dwóch wektorów. Dla `a` o kształcie `(M,)` i `b` o kształcie `(K+1,)`:

```
np.outer(a, b)[i, j] = a[i] * b[j]
```

Wynik: macierz o kształcie `(M, K+1)`.

### Dlaczego `outer`, a nie zwykłe `*`?

Bo **chcemy gradient o kształcie `(M, K+1)` — dokładnie taki sam jak `W2`**.

Dla `delta_output` o kształcie `(M,)` i `v` o kształcie `(K+1,)`:

```
delta_output * v          → ERROR jeśli M ≠ K+1
                            (lub broadcast jeśli pasuje, ale nie to chcemy)

np.outer(delta_output, v) → kształt (M, K+1) ✓
```

### Co dokładnie liczy `outer`?

```
dW2[m, k] = delta_output[m] · v[k]
          = δ_out_m · v_k
          = ∂E/∂w⁽²⁾_mk
```

🔍 **Dokładnie wzór z części 3.4!** Każdy element `dW2[m, k]` to gradient dla wagi `w⁽²⁾_mk`. `outer` jednocześnie liczy **wszystkie 3 gradienty W²** (M=1, K+1=3).

---

## 4.7 Delta warstwy ukrytej — NAJTRUDNIEJSZA LINIA

```python
delta_hidden = (W2[:, 1:].T @ delta_output) * tanh_derivative(v_hidden, beta)  # (K,)
```

Rozłóżmy to na trzy kawałki.

### Kawałek 1: `W2[:, 1:]`

```python
W2[:, 1:]  # kształt (M, K) — wszystkie wiersze, kolumny od 1 do końca
```

🔍 **Dlaczego `[:, 1:]`, a nie cały `W2`?**

`W2` ma kształt `(M, K+1)`. **Pierwsza kolumna `W2[:, 0]` to wagi BIASU.** Bias nie pochodzi od żadnego neuronu warstwy ukrytej — to stała `1`. Więc nie ma sensu propagować błędu do "neuronu biasu" (bo go nie ma).

`W2[:, 1:]` = wagi tylko dla rzeczywistych neuronów ukrytych. Kształt `(M, K)`.

### Kawałek 2: `W2[:, 1:].T @ delta_output`

```python
W2[:, 1:].T          # kształt (K, M)
delta_output         # kształt (M,)
W2[:, 1:].T @ delta_output  # kształt (K,)
```

🔍 **To liczy `Σ_m δ_out_m · w⁽²⁾_mk` dla każdego k.**

```
result[k] = Σ_m W2[m, k+1] · delta_output[m]
          = Σ_m w⁽²⁾_mk · δ_out_m
```

🔍 **Dlaczego transponujemy `.T`?**

Bo chcemy mnożyć **wzdłuż osi M** (sumować po m). Bez transpozycji `(M, K) @ (M,)` to błąd kształtów. Z transpozycją `(K, M) @ (M,) → (K,)`. To wektor "błędów propagowanych do każdego neuronu ukrytego".

### Kawałek 3: `* tanh_derivative(v_hidden, beta)`

```python
tanh_derivative(v_hidden, beta)  # kształt (K,) — pochodna tanh dla każdego neuronu
```

🔍 **Element-wise mnożenie** dwóch wektorów `(K,)`:

```
delta_hidden[k] = (Σ_m δ_out_m · w⁽²⁾_mk) · β · (1 - v_hidden_k²)
                  └────────────────────┘   └──────────────┘
                  błąd propagowany           pochodna tanh
```

🔍 **Dokładnie δ_hidden z części 3.5!**

---

## 4.8 Gradient W¹

```python
dW1 += np.outer(delta_hidden, x)  # (K, N+1)
```

🔍 **Identyczny wzór jak dla W²:** `outer(delta_warstwy, wejście)`.

```
dW1[k, n] = delta_hidden[k] · x[n]
          = δ_hidden_k · x_n
          = ∂E/∂w⁽¹⁾_kn
```

🔍 **Tu używamy `x`, a nie `v`.** Bo wejściem do warstwy ukrytej jest właśnie wektor `x` z biasem.

---

## 4.9 Aktualizacja wag (po epoce)

```python
W1 -= eta * dW1
W2 -= eta * dW2
```

🔍 **`-=` to ten sam wzór gradient descent:**

```
W = W - η · dW
```

Aktualizujemy **dopiero po epoce** (poza pętlą po próbkach), bo `dW1` i `dW2` zostały zsumowane przez wszystkie próbki.

🔍 **Dla wariantu `train_sample` (po próbce) ta linia byłaby WEWNĄTRZ pętli po próbkach**, a `dW1` / `dW2` byłyby zerowane przed każdą próbką (lub w ogóle nie używane jako akumulatory — tylko liczone bezpośrednio).

---

<a id="5"></a>

# 5. Tablica kształtów — który wymiar gdzie

> **Najczęstsze źródło błędów: nieprawidłowe kształty. Tutaj wszystko w jednym miejscu.**

## 5.1 Notacja

- **N** = liczba wejść (dla XOR: N=2)
- **K** = liczba neuronów ukrytych (dla zadania: K=2)
- **M** = liczba neuronów wyjściowych (dla XOR: M=1)

## 5.2 Wszystkie kształty

| Symbol | Kształt | Znaczenie |
|--------|---------|-----------|
| `x` | `(N+1,)` = `(3,)` | wejście z biasem `[1, x1, x2]` |
| `W1` | `(K, N+1)` = `(2, 3)` | wagi warstwy ukrytej |
| `z_hidden` | `(K,)` = `(2,)` | pre-aktywacja ukrytej |
| `v_hidden` | `(K,)` = `(2,)` | post-aktywacja ukrytej |
| `v` | `(K+1,)` = `(3,)` | post-aktywacja z biasem |
| `W2` | `(M, K+1)` = `(1, 3)` | wagi warstwy wyjściowej |
| `z_output` | `(M,)` = `(1,)` | pre-aktywacja wyjściowej |
| `y` | `(M,)` = `(1,)` | wynik sieci |
| `target` | `(M,)` = `(1,)` | oczekiwana odpowiedź |
| `error` | `(M,)` = `(1,)` | `y - d` |
| `delta_output` | `(M,)` = `(1,)` | δ warstwy wyjściowej |
| `delta_hidden` | `(K,)` = `(2,)` | δ warstwy ukrytej |
| `dW1` | `(K, N+1)` = `(2, 3)` | gradient W1 |
| `dW2` | `(M, K+1)` = `(1, 3)` | gradient W2 |

## 5.3 Reguła kciuka

🔍 **Gradient ma ZAWSZE ten sam kształt co wagi.** Jeśli liczysz `dW1` i wychodzi inny kształt niż `W1` — masz błąd.

🔍 **`outer(delta, input)` produkuje gradient.** `delta` ma długość = liczba neuronów tej warstwy. `input` ma długość = liczba wejść do tej warstwy (z biasem).

---

<a id="6"></a>

# 6. Pułapki numeryczne i decyzje projektowe

## 6.1 Dlaczego XOR z `[-1, 1]` zamiast `[0, 1]`?

W zadaniu 2 instrukcja każe zamienić:

```python
# zle:
xx = np.array([[1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])

# dobrze:
xx = np.array([[1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1]])
```

🔍 **Dlaczego?**

Spójrz na gradient dla W1:

```
∂E/∂w⁽¹⁾_kn = δ_hidden_k · x_n
```

Jeśli `x_n = 0`, to **gradient = 0**. Niezależnie od błędu, niezależnie od delty. Ta waga **nigdy się nie zaktualizuje**. To są **martwe gradienty**.

Z `[0, 0]` jako wejściem mamy 4 z 9 wag wejściowych martwych w tej próbce. Sieć utyka.

Z `[-1, -1]` zamiast `[0, 0]`:
- gradient `δ · (-1) = -δ` ≠ 0
- waga się aktualizuje
- sieć się uczy

🔍 **Reguła:** unikaj zerowych wejść w problemach klasyfikacji binarnej, jeśli masz wagi-jako-bias.

---

## 6.2 Dlaczego `eta=2.0` w demo, a nie `0.01`?

Notebook używa `eta=2.0`. Brzmi dużo, prawda? Standardowo widzi się `0.01` lub `0.001`.

🔍 **XOR to bardzo mały problem.** 4 próbki, 9 wag, prosty krajobraz błędu. Można sobie pozwolić na duże kroki.

🔍 **W realnych problemach** (CNN, transformery) używa się `lr=0.001` lub mniej, bo:
- jest miliony wag
- krajobraz błędu jest skomplikowany
- za duży krok = oscylacje, sieć nie zbiega

🔍 **W zadaniu 2 eksperymentuj z `eta`.** Spróbuj `0.1`, `0.5`, `1.0`, `5.0`. Zobacz jak zmienia się krzywa uczenia. To dobra intuicja.

---

## 6.3 Dlaczego błąd "skacze" zamiast monotonicznie spadać?

Krzywa uczenia często wygląda tak:

```
błąd
 │
 │  \
 │   \___
 │       \      ╱╲    ← skoki!
 │        \____╱  ╲__
 │                   \
 │                    \____
 └───────────────────────── epoki
```

🔍 **Powody:**

1. **Wariant online (po próbce)** — każda próbka ciągnie wagi w swoją stronę, sąsiednie próbki mogą "kłócić się"
2. **Za duże eta** — przeskakujemy minimum, oscylujemy
3. **Plateau przed przełomem** — sieć utknęła w "siodle", potem nagle znajduje wyjście

🔍 **Skoki NIE OZNACZAJĄ buga.** Oznaczają, że uczenie jest "zaszumione". Dla XOR to normalne. W realnych zadaniach używa się momentum/Adam żeby to wygładzić (Lab 4).

---

## 6.4 Dlaczego `np.random.seed(42)`?

```python
np.random.seed(42)
```

🔍 **Powtarzalność.** Bez seed za każdym razem dostaniesz inne losowe wagi → różne krzywe uczenia → trudno debugować.

Z seed: zawsze te same wagi początkowe → te same wyniki → możesz porównywać eksperymenty.

🔍 **Dlaczego 42?** [Konwencja z "Hitchhiker's Guide to the Galaxy"](https://en.wikipedia.org/wiki/42_(number)). Magic number wśród programistów.

---

<a id="7"></a>

# 7. Zadanie 1 — rozwiązanie z komentarzem

> **Komórka 21 w notebooku ma `pass` — tu masz pełne rozwiązanie z wyjaśnieniem każdej linii.**

## 7.1 Pełne rozwiązanie

```python
import numpy as np

def sigmoid(x, beta):
    return 1.0 / (1.0 + np.exp(-beta * x))

def tanh(x, beta):
    return np.tanh(beta * x)

def mlp(x, w1, w2, beta):
    """
    Forward pass MLP 2-2-1.
    
    x:    np.array (3,) — wejście z biasem [1, x1, x2]
    w1:   np.array (2, 3) — wagi warstwy ukrytej
    w2:   np.array (1, 3) — wagi warstwy wyjściowej
    beta: float — parametr aktywacji
    
    Zwraca: (y, v, v_hidden)
    """
    # === Warstwa ukryta ===
    z_hidden = w1 @ x                       # (2,3) @ (3,) → (2,)
    v_hidden = tanh(z_hidden, beta)         # element-wise tanh → (2,)
    
    # === Bias dla warstwy wyjściowej (Protip 1) ===
    v = np.concatenate([[1], v_hidden])     # (3,) → [1, v1, v2]
    
    # === Warstwa wyjściowa ===
    z_out = w2 @ v                          # (1,3) @ (3,) → (1,)
    y = sigmoid(z_out, beta)                # (1,) → (1,) w (0,1)
    
    return y, v, v_hidden
```

## 7.2 Test

```python
np.random.seed(0)
w1_test = np.random.randn(2, 3) * 0.5
w2_test = np.random.randn(1, 3) * 0.5

y, v, v_hidden = mlp(np.array([1, 0, 1]), w1_test, w2_test, beta=1.0)
print(f"y = {y}")              # spodziewane: jakiś float w (0,1)
print(f"v = {v}")              # spodziewane: [1, v1, v2]
print(f"v_hidden = {v_hidden}")# spodziewane: [v1, v2] w (-1,1)
```

## 7.3 Co ZWRACAMY i dlaczego

| Zmienna | Po co | Używamy w |
|---------|-------|-----------|
| `y` | Główny wynik sieci | porównanie z target, klasyfikacja |
| `v` | Wejście do warstwy wyjściowej (z biasem) | `dW2 = outer(delta_out, v)` w backprop |
| `v_hidden` | Aktywacje ukrytej (bez biasu) | `tanh_diff(v_hidden, beta)` w backprop |

🔍 **Bez `v` i `v_hidden` nie zaimplementujesz backpropagation.** Dlatego instrukcja mówi: "funkcja powinna zwracać odpowiedź sieci ORAZ wszystkie składniki potrzebne do obliczania pochodnych".

## 7.4 Walidacja kształtów (opcjonalnie)

```python
def mlp(x, w1, w2, beta):
    # Walidacja (Protip 2)
    assert x.ndim == 1, f"x powinien być 1D, ma {x.ndim}D"
    assert w1.shape[1] == x.shape[0], \
        f"Niezgodność: w1 ma {w1.shape[1]} kolumn, x ma {x.shape[0]} elementów"
    assert w2.shape[1] == w1.shape[0] + 1, \
        f"Niezgodność: w2 powinien mieć {w1.shape[0]+1} kolumn (K+1), ma {w2.shape[1]}"
    
    # ... reszta jak wyżej
```

🔍 **Dlaczego asserty?** Bo numpy często **wycisza** błędy kształtów przez broadcasting. Dostajesz dziwne wyniki, ale bez wyjątku. Lepiej jawnie sprawdzić.

---

<a id="8"></a>

# 8. Zadanie 2 — rozwiązanie z komentarzem

> **Komórka 23 w notebooku ma szkielety z `pass` — tu masz pełne rozwiązanie obu wariantów.**

## 8.1 Wariant 1: `train_sample` (aktualizacja po próbce)

```python
def train_sample(xx, d, eta, beta, max_epochs=100_000):
    """
    Trening MLP z aktualizacją wag PO KAŻDEJ próbce (online/stochastic).
    
    xx: dane wejściowe z biasem, kształt (4, 3) dla XOR
    d:  oczekiwane wyjścia, kształt (4,)
    """
    np.random.seed(42)
    
    # Inicjalizacja wag (małe losowe)
    w1 = np.random.randn(2, 3) * 0.5  # (K, N+1) = (2, 3)
    w2 = np.random.randn(1, 3) * 0.5  # (M, K+1) = (1, 3)
    
    errors = []
    
    for epoch in range(max_epochs):
        epoch_error = 0.0
        misclassified = 0
        
        for i in range(len(xx)):
            x = xx[i]              # (3,) — już z biasem
            target = d[i]          # skalar
            
            # === FORWARD ===
            y, v, v_hidden = mlp(x, w1, w2, beta)
            # y: (1,), v: (3,), v_hidden: (2,)
            
            # === BŁĄD ===
            error = y - target     # (1,)
            epoch_error += 0.5 * np.sum(error**2)
            
            # Klasyfikacja (zasady z zadania)
            if y[0] > 0.9:
                cls = 1
            elif y[0] < 0.1:
                cls = 0
            else:
                cls = -1  # nieokreślona
            if cls != target:
                misclassified += 1
            
            # === BACKPROP ===
            
            # Delta wyjściowa
            delta_out = error * sigmoid_diff(y, beta)  # (1,)
            
            # Gradient W2
            dw2 = np.outer(delta_out, v)               # (1, 3)
            
            # Delta ukryta (uwaga na [:, 1:] — pomijamy bias!)
            delta_hidden = (w2[:, 1:].T @ delta_out) * tanh_diff(v_hidden, beta)  # (2,)
            
            # Gradient W1
            dw1 = np.outer(delta_hidden, x)            # (2, 3)
            
            # === AKTUALIZACJA OD RAZU (online) ===
            w1 -= eta * dw1
            w2 -= eta * dw2
        
        errors.append(epoch_error)
        
        # Stop jeśli wszystkie próbki klasyfikują się poprawnie
        if misclassified == 0:
            print(f"Zbiezność po {epoch+1} epokach")
            break
    
    return w1, w2, errors
```

🔍 **Kluczowa cecha:** `w1 -= eta * dw1` jest **WEWNĄTRZ pętli po próbkach**. Każda próbka od razu aktualizuje wagi.

🔍 **Konsekwencje:**
- Większy szum w uczeniu (każda próbka ciągnie wagi w swoją stronę)
- Może uciec z minimum lokalnego (szum jest pomocny)
- Szybsza konwergencja na prostych problemach

---

## 8.2 Wariant 2: `train_epoch` (aktualizacja po epoce)

```python
def train_epoch(xx, d, eta, beta, max_epochs=100_000):
    """
    Trening MLP z akumulacją gradientów i aktualizacją PO EPOCE (batch).
    """
    np.random.seed(42)
    
    w1 = np.random.randn(2, 3) * 0.5
    w2 = np.random.randn(1, 3) * 0.5
    
    errors = []
    
    for epoch in range(max_epochs):
        epoch_error = 0.0
        misclassified = 0
        
        # === AKUMULATORY (Protip 3) ===
        dw1_acc = np.zeros_like(w1)
        dw2_acc = np.zeros_like(w2)
        
        for i in range(len(xx)):
            x = xx[i]
            target = d[i]
            
            y, v, v_hidden = mlp(x, w1, w2, beta)
            
            error = y - target
            epoch_error += 0.5 * np.sum(error**2)
            
            if y[0] > 0.9: cls = 1
            elif y[0] < 0.1: cls = 0
            else: cls = -1
            if cls != target:
                misclassified += 1
            
            # Backprop — liczymy gradient
            delta_out = error * sigmoid_diff(y, beta)
            delta_hidden = (w2[:, 1:].T @ delta_out) * tanh_diff(v_hidden, beta)
            
            # AKUMULUJEMY (nie aktualizujemy wag)
            dw2_acc += np.outer(delta_out, v)
            dw1_acc += np.outer(delta_hidden, x)
        
        # === AKTUALIZACJA RAZ NA EPOKĘ ===
        w1 -= eta * dw1_acc
        w2 -= eta * dw2_acc
        
        errors.append(epoch_error)
        
        if misclassified == 0:
            print(f"Zbiezność po {epoch+1} epokach")
            break
    
    return w1, w2, errors
```

🔍 **Różnice względem `train_sample`:**

| Aspekt | `train_sample` | `train_epoch` |
|--------|----------------|---------------|
| Akumulatory | Brak | `dw1_acc`, `dw2_acc` |
| Aktualizacja wag | Po każdej próbce | Po pętli przez wszystkie próbki |
| Zbieżność | Szybsza, ale szumna | Wolniejsza, ale stabilna |
| Minima lokalne | Może uciec dzięki szumowi | Może utknąć |

---

## 8.3 Porównanie + wykres

```python
import matplotlib.pyplot as plt

xx = np.array([[1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1]])
d = np.array([0, 1, 1, 0])

w1_s, w2_s, errors_sample = train_sample(xx, d, eta=0.5, beta=1.0)
w1_e, w2_e, errors_epoch = train_epoch(xx, d, eta=0.5, beta=1.0)

plt.figure(figsize=(10, 5))
plt.plot(errors_sample, label='Po próbce (online)', color='#E63888', alpha=0.8)
plt.plot(errors_epoch, label='Po epoce (batch)', color='#1B1E3D', alpha=0.8)
plt.xlabel('Epoka')
plt.ylabel('Błąd MSE')
plt.title('XOR — porównanie wariantów aktualizacji wag')
plt.legend()
plt.yscale('log')
plt.grid(alpha=0.3)
plt.show()
```

🔍 **Co zobaczysz:**

- **Krzywa "po próbce"** zwykle zbiega szybciej (mniej epok), ale jest szumna
- **Krzywa "po epoce"** zbiega wolniej, ale gładko
- Obie powinny dojść do ~10⁻⁴ lub niżej jeśli sieć faktycznie nauczyła się XOR

---

<a id="9"></a>

# 9. Odpowiedzi na pytania kontrolne

> **Komórka 24 w notebooku zadaje pytania kontrolne. Tu masz wzorcowe odpowiedzi.**

## 9.1 Czym różni się MLP od pojedynczego perceptronu?

| Cecha | Perceptron | MLP |
|-------|------------|-----|
| Liczba warstw | 1 | 2+ |
| Funkcja aktywacji | Progowa (step) | Ciągła (sigmoid, tanh) |
| Granica decyzyjna | Liniowa (prosta) | Nieliniowa (dowolna) |
| XOR? | NIE | TAK |
| Uczenie | Reguła perceptronu | Backpropagation |
| Różniczkowalność | Brak | Pełna (wszędzie) |

**Najważniejsza różnica:** MLP ma warstwę ukrytą, dzięki czemu może modelować dowolne nieliniowe zależności. To wynika z **uniwersalnego twierdzenia o aproksymacji** (Cybenko, 1989).

---

## 9.2 Dlaczego używamy sigmoid/tanh zamiast funkcji progowej?

**Trzy powody:**

1. **Różniczkowalność** — funkcja progowa nie ma pochodnej w punkcie skoku. Bez pochodnej nie ma gradient descent, nie ma backpropagation.

2. **Ciągłość gradientu** — pochodne sigmoid/tanh są ciągłe, więc małe zmiany wag dają małe zmiany wyjścia. To stabilizuje uczenie.

3. **Wartości pośrednie** — zamiast tylko 0/1, sieć zwraca wartość pośrednią (np. 0.7 = "raczej tak, ale nie do końca"). To umożliwia stopniowe uczenie i interpretację probabilistyczną.

---

## 9.3 Co robi forward pass, a co backpropagation?

| Faza | Co robi | Kierunek przepływu |
|------|---------|---------------------|
| Forward pass | Liczy wyjście sieci dla danego wejścia | wejście → wyjście |
| Backpropagation | Liczy gradient błędu względem każdej wagi | wyjście → wejście |
| Update | Aktualizuje wagi: `w -= η · ∂E/∂w` | wszystkie wagi |

**Cykl uczenia:**

```
forward → loss → backward → update → forward → ...
```

Każda iteracja to jedna próba zmniejszenia błędu o "η razy gradient".

---

## 9.4 Dlaczego potrzebujemy pochodnych funkcji aktywacyjnych?

Bo **gradient błędu względem wag idzie PRZEZ funkcje aktywacji** w łańcuchu pochodnych:

```
∂E/∂w⁽¹⁾ = ... · σ'(z_out) · ... · tanh'(z_hidden) · ...
                 └────────┘         └─────────────┘
                pochodne aktywacji są PIERWSZORZĘDNYM elementem gradientu
```

Bez pochodnych aktywacji **łańcuch reguły łańcuchowej się rwie** — nie możesz policzyć gradientu, nie możesz aktualizować wag.

🔍 **Dlatego sigmoid i tanh są tak popularne** — ich pochodne są:
- ciągłe
- proste do obliczenia (`σ·(1-σ)` lub `1-tanh²`)
- wyrażone przez wartość funkcji (oszczędność obliczeń)

---

## 9.5 Jaka jest różnica między aktualizacją wag po próbce a po epoce?

| Aspekt | Po próbce (online/SGD) | Po epoce (batch) |
|--------|------------------------|------------------|
| Kiedy aktualizujesz | Po każdej próbce | Po przejściu wszystkich próbek |
| Liczba aktualizacji na epokę | Tyle co próbek | 1 |
| Wariancja gradientu | Wysoka (szum) | Niska (uśredniona) |
| Pamięć | Mniej (jeden gradient) | Więcej (akumulator) |
| Konwergencja na prostych zadaniach | Szybsza | Wolniejsza |
| Konwergencja na trudnych zadaniach | Niestabilna | Stabilna |
| Minima lokalne | Może uciec (szum pomaga) | Może utknąć |

🔍 **W praktyce:** używa się **mini-batch** — kompromis. Aktualizujesz po np. 32 próbkach. To standard w deep learning.

---

## 9.6 Dlaczego zerowe wejścia XOR (0,0) warto zamienić na (-1,-1)?

**Bo gradient `∂E/∂w⁽¹⁾_kn` zawiera `x_n` jako mnożnik:**

```
∂E/∂w⁽¹⁾_kn = δ_hidden_k · x_n
```

Jeśli `x_n = 0`:
```
∂E/∂w⁽¹⁾_kn = δ_hidden_k · 0 = 0
```

**Gradient = 0 → waga się nie zmienia → sieć utyka.**

Z `x_n = -1`:
```
∂E/∂w⁽¹⁾_kn = δ_hidden_k · (-1) = -δ_hidden_k ≠ 0
```

Waga się aktualizuje, sieć się uczy.

🔍 **To "martwe gradienty" w skali mikro.** W większych sieciach z ReLU mamy podobny problem ("dying ReLU") — neurony, które dają 0 dla wszystkich wejść, nigdy się nie uczą.

---

# 📚 Mapa: który fragment notebooka tłumaczy który fragment tego pliku

| Komórka notebooka | Sekcja w tym pliku |
|---|---|
| 6 (definicja sigmoid_derivative, tanh_derivative) | [1. Wyprowadzenie pochodnych](#1) |
| 10 (DEMO forward pass krok po kroku) | [2. Anatomia forward pass](#2) |
| 11 (markdown z wzorami gradientów) | [3. Reguła łańcuchowa](#3) |
| 13 (mlp_forward) | [2.2 Linia po linii](#2) |
| 14 (train_mlp_xor — backprop) | [4. Backpropagation w kodzie](#4) |
| 21 (Zadanie 1 — szkielet) | [7. Rozwiązanie Z1](#7) |
| 23 (Zadanie 2 — szkielet) | [8. Rozwiązanie Z2](#8) |
| 24 (pytania kontrolne) | [9. Odpowiedzi](#9) |

---

# ✅ Checklist studenta przed oddaniem zadania

- [ ] Funkcja `mlp()` zwraca **trzy wartości**: `y`, `v`, `v_hidden`
- [ ] W `mlp()` używasz `np.concatenate([[1], v_hidden])` dla biasu warstwy wyjściowej
- [ ] W backpropie używasz `W2[:, 1:].T` (nie całego W2!) przy liczeniu `delta_hidden`
- [ ] Używasz `np.outer(delta, input)` do liczenia gradientów wag
- [ ] Dane XOR mają `[1, -1, -1]` dla pierwszej próbki, nie `[1, 0, 0]`
- [ ] Inicjalizujesz wagi małymi losowymi wartościami (`* 0.5` lub mniej)
- [ ] Używasz `np.random.seed(42)` dla powtarzalności
- [ ] Sprawdzasz klasyfikację: `y > 0.9 → 1`, `y < 0.1 → 0`, inaczej `nieokreślona`
- [ ] Wykres błędu w skali logarytmicznej (`plt.yscale('log')`)
- [ ] Porównujesz oba warianty (`train_sample` vs `train_epoch`) na jednym wykresie

---

**Powodzenia!** Jak coś jest niejasne — wracaj do tego pliku, każda sekcja ma odpowiednią komórkę w notebooku.