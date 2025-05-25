# Kursinis

## Projekto vykdymas

Projekto vykdymas susideda iš šių žingsnių:

1. **Duomenų aibės sudarymas**  
2. **Įterpinių modelio atranka ir testavimas**  
3. **Funkcijų vertimas į įterpinius naudojant pasirinktą modelį**  
4. **Vektorinės duomenų bazės konteinerio paleidimas**  
5. **Vektorinės duomenų bazės testavimas ir rezultatų analizė**  

---

### 1. Duomenų aibės sudarymas

Visų pirma paleidžiamas `.py` failas duomenų aibės sudarymui. Šis skriptas surenka ir paruošia visus reikalingus duomenis analizėms.

### 2. Įterpinių modelio atranka ir testavimas

Lygiagrečiai vykdomi skirtingų įterpinių (embedding) modelių testavimai. Įvertiname kiekvieno modelio tikslumą ir našumą, siekdami pasirinkti geriausią.

### 3. Funkcijų vertimas į įterpinius

Gautą duomenų aibę perkeliame į HPC aplinką. Joje naudojami šie failai:

- `embed_functions.py` – funkcijų įterpčių generavimo modulis  
- `run_embed.sh` – paleidimo skriptas, automatiškai kviečiantis `embed_functions.py`

Paleidę `run_embed.sh`, sukuriami įterpiniai (embeddings) kiekvienai duomenų aibės funkcijai.

### 4. Vektorinės duomenų bazės konteinerio paleidimas

Paruoštą įterpinių lentelę iš HPC aplinkos perkeliame į atskirą vektorinės duomenų bazės konteinerį. Čia sukuriamas Docker (ar kito tipo) konteineris, kuriame veikia pasirinkta vektorinė duomenų bazė.
