#!/usr/bin/env python3
"""
Preflight check & download — uruchom RAZ przed otwarciem notebooka.

Zapewnia że wszystkie potrzebne datasety i modele są pobrane lokalnie.

Użycie:
    python preflight_download.py

Co robi:
1. Sprawdza wymagane pakiety Pythona (tensorflow, datasets, sklearn, ...)
2. Pobiera 3 datasety z Hugging Face Hub:
   - zalando-datasets/fashion_mnist (~30MB)
   - uoft-cs/cifar10 (~170MB)
   - dair-ai/emotion (~2MB)
3. Wczytuje iris z scikit-learn (lokalny, bez downloadu)
4. Sanity check kazdego

Przy pierwszym uruchomieniu: ~5-10 min (download CIFAR-10 dominuje).
Przy kolejnych: < 30s (wszystko z cache).
"""
import os, sys, time

os.environ["HF_DATASETS_DISABLE_PROGRESS_BAR"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

REQUIRED_PACKAGES = {
    "tensorflow": "tensorflow",
    "datasets": "datasets",
    "sklearn": "scikit-learn",
    "matplotlib": "matplotlib",
    "numpy": "numpy",
}


def check_packages():
    print("=== 1/4 Sprawdzanie pakietow ===")
    missing = []
    for module, pip_name in REQUIRED_PACKAGES.items():
        try:
            __import__(module)
            print(f"  [OK]   {module}")
        except ImportError:
            print(f"  [FAIL] {module} -- zainstaluj: pip install {pip_name}")
            missing.append(pip_name)
    if missing:
        print(f"\nBrakuje {len(missing)} pakietow. Zainstaluj:")
        print(f"  pip install {' '.join(missing)}")
        sys.exit(1)


def download_hf_datasets():
    print("\n=== 2/4 Pobieranie datasetow z HF Hub ===")
    from datasets import load_dataset

    datasets_to_pull = [
        ("zalando-datasets/fashion_mnist", "~30MB"),
        ("uoft-cs/cifar10", "~170MB -- to potrwa najdluzej"),
        ("dair-ai/emotion", "~2MB"),
    ]
    fetched = {}
    for name, size in datasets_to_pull:
        print(f"\n  -> {name} ({size})")
        t0 = time.time()
        try:
            ds = load_dataset(name)
            elapsed = time.time() - t0
            sizes = {split: len(ds[split]) for split in ds.keys()}
            print(f"     OK ({elapsed:.1f}s) | splits: {sizes}")
            fetched[name] = ds
        except Exception as e:
            print(f"     FAIL: {type(e).__name__}: {e}")
            sys.exit(1)
    return fetched


def check_sklearn_iris():
    print("\n=== 3/4 Sprawdzanie sklearn iris (lokalny) ===")
    from sklearn.datasets import load_iris
    iris = load_iris()
    print(f"  iris: {iris.data.shape}, klasy: {iris.target_names.tolist()}")
    print(f"  cechy: {iris.feature_names}")


def sanity_check_keys(fetched):
    print("\n=== 4/4 Sanity check struktury danych ===")
    for name, ds in fetched.items():
        split = list(ds.keys())[0]
        sample = ds[split][0]
        print(f"  {name}/{split}[0] keys: {list(sample.keys())}")
    # CIFAR-10 ma kolumne 'img' (nie 'image'), Fashion ma 'image'.
    # Ta sanity check jest istotna -- notebook polega na tej roznicy.


if __name__ == "__main__":
    print("=" * 60)
    print("Preflight check — BAI zestaw zaliczeniowy")
    print("=" * 60)
    check_packages()
    fetched = download_hf_datasets()
    check_sklearn_iris()
    sanity_check_keys(fetched)
    print("\n" + "=" * 60)
    print("OK — mozesz otworzyc BAI_Zestaw_Zaliczeniowy.ipynb")
    print("=" * 60)
