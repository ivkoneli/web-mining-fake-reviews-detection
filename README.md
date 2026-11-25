# Prepoznavanje lažnih recenzija proizvoda i usluga

Projekat iz predmeta Web mining (tema 20).

Lažne recenzije, pozitivne ili negativne, namerno obmanjuju kupce. Projekat ih otkriva na osnovu tri izvora signala:

- **tekst recenzije**: dužina, TF-IDF n-grami, sentiment, zamenice u prvom licu, interpunkcija
- **ponašanje autora**: broj recenzija, odstupanje ocene od proseka proizvoda, udeo ekstremnih ocena (1★ i 5★), broj recenzija u jednom danu
- **obrasci objavljivanja**: naleti recenzija na istom proizvodu, recenzije napisane odmah posle otvaranja naloga

## Ciljevi

1. Pronaći obeležja po kojima se lažne recenzije razlikuju od pravih (EDA).
2. Napraviti klasifikator lažno/pravo.
3. Proceniti koliko lažne recenzije menjaju prosečnu ocenu proizvoda, poređenjem proseka sa njima i bez njih.
4. Proveriti da li model trenira na jednom skupu i radi na drugom (npr. YelpChi → YelpNYC, Ott → Yelp).
5. Utvrditi koja obeležja su najvažnija (permutaciona važnost / SHAP).

## Metrike

ROC-AUC, average precision (AP), F1 i Precision@k

## Podaci

| Skup | Opis | Oznaka |
|---|---|---|
| YelpChi | recenzije hotela i restorana u Čikagu | Yelp-ov filter (filtrirano = lažno) |
| YelpNYC | recenzije restorana u Njujorku | Yelp-ov filter |
| YelpZip | restorani u više američkih saveznih država (po poštanskom broju) | Yelp-ov filter |
| Ott et al. | 1.600 hotelskih recenzija, polovina namerno lažna (pisane preko MTurk-a) | gold standard |

Izvor: https://odds.cs.stonybrook.edu/?p=562

Sirovi podaci nisu u repozitorijumu (`data/` je u `.gitignore`).

## Struktura

```
data/        # skupovi podataka (lokalno)
notebooks/   # analiza i modeli
```
