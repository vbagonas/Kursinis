# Kursinis

## Projekto vykdymas

Projekto vykdymas susideda iš šių žingsnių:

1. **Duomenų aibės sudarymas**  
2. **Įterpinių modelio atranka ir testavimas**  
3. **Funkcijų vertimas į įterpinius naudojant pasirinktą modelį**  
4. **Vektorinės duomenų bazės konteinerio paleidimas**  
5. **Vektorinės duomenų bazės testavimas**  

---

### 1. Duomenų aibės sudarymas

Su `1_kodu_filtravimas.py` ir `2_funkciju_istraukimas.py` failais buvo sukurta 100 tūkstančių funkcijų duomenų aibė.

### 2. Įterpinių modelio atranka ir testavimas

Lygiagrečiai vykdomi skirtingų įterpinių (embedding) modelių testavimai. Įvertiname kiekvieno modelio tikslumą ir našumą, siekdami pasirinkti geriausią. Šiai užduočiai atlikti naudojami šie failai: `modeliu_palyginimas.sh`, `modeliu_palyginimas_su_klonu_uzduotimi.py`, `klonu_uzd_duomenu_rinkinio_sudarymas.py`, `iterpiniu_vizualizavimas_umap.py`. Duomenys naudojami šiems Python failams įkelti kaip .zip

### 3. Funkcijų vertimas į įterpinius

Gautą duomenų aibę perkeliame į HPC aplinką. Joje naudojami šie failai:

- `embed_functions.py` – funkcijų įterpinių generavimo modelis  
- `run_embed.sh` – paleidimo skriptas, automatiškai kviečiantis `embed_functions.py`

Paleidę `run_embed.sh`, sukuriami įterpiniai (angl. embeddings) kiekvienai duomenų aibės funkcijai.

### 4. Vektorinės duomenų bazės konteinerio paleidimas

Terminale paleidžiamas `docker-compose.yml` failas, kuris Docker aplinkoje sukuria konteinerį.

### 5. Vektorinės duomenų bazės testavimas

Su `Milvus_sukurimas_testavimas.py` yra sukuriama kolekcija, kurioje yra talpinami .parquet failų duomenys (įterpiniai su jų metaduomenimis). Toliau atliekami indeksų ir panašumo metrikų testavimai

