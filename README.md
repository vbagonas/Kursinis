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

Visų pirma paleidžiamas `.py` failas duomenų aibės sudarymui. Šis skriptas surenka ir paruošia visus reikalingus duomenis analizėms.

### 2. Įterpinių modelio atranka ir testavimas

Lygiagrečiai vykdomi skirtingų įterpinių (embedding) modelių testavimai. Įvertiname kiekvieno modelio tikslumą ir našumą, siekdami pasirinkti geriausią.

### 3. Funkcijų vertimas į įterpinius

Gautą duomenų aibę perkeliame į HPC aplinką. Joje naudojami šie failai:

- `embed_functions.py` – funkcijų įterpinių generavimo modelis  
- `run_embed.sh` – paleidimo skriptas, automatiškai kviečiantis `embed_functions.py`

Paleidę `run_embed.sh`, sukuriami įterpiniai (angl. embeddings) kiekvienai duomenų aibės funkcijai.

### 4. Vektorinės duomenų bazės konteinerio paleidimas

Terminale paleidžiamas `docker-compose.yml` failas, kuris Docker aplinkoje sukuria konteinerį.

### 5. Vektorinės duomenų bazės testavimas

Su `Milvus_sukurimas_testavimas.py` yra sukuriama kolekcija, kurioje yra talpinami įterpiniai su jų metaduomenimis. Toliau atliekami indeksų ir panašumo metrikų testavimai

